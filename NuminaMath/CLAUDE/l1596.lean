import Mathlib

namespace NUMINAMATH_CALUDE_ratio_problem_l1596_159616

/-- Given two ratios a:b:c and c:d:e, prove that a:e is 3:10 -/
theorem ratio_problem (a b c d e : ℚ) 
  (h1 : a / b = 2 / 3 ∧ b / c = 3 / 4)
  (h2 : c / d = 3 / 4 ∧ d / e = 4 / 5) :
  a / e = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1596_159616


namespace NUMINAMATH_CALUDE_denarii_puzzle_l1596_159680

theorem denarii_puzzle (x y : ℚ) : 
  (x + 7 = 5 * (y - 7)) →
  (y + 5 = 7 * (x - 5)) →
  (x = 11 + 9 / 17 ∧ y = 9 + 14 / 17) :=
by sorry

end NUMINAMATH_CALUDE_denarii_puzzle_l1596_159680


namespace NUMINAMATH_CALUDE_log_equation_solution_l1596_159612

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → log 4 (x^3) + log (1/4) x = 12 → x = 4096 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1596_159612


namespace NUMINAMATH_CALUDE_max_profit_is_21000_l1596_159631

/-- Represents the production capabilities and constraints of a furniture factory -/
structure FurnitureFactory where
  carpenterHoursChair : ℕ
  carpenterHoursDesk : ℕ
  maxCarpenterHours : ℕ
  painterHoursChair : ℕ
  painterHoursDesk : ℕ
  maxPainterHours : ℕ
  profitChair : ℕ
  profitDesk : ℕ

/-- Calculates the profit for a given production plan -/
def calculateProfit (factory : FurnitureFactory) (chairs : ℕ) (desks : ℕ) : ℕ :=
  chairs * factory.profitChair + desks * factory.profitDesk

/-- Checks if a production plan is feasible given the factory's constraints -/
def isFeasible (factory : FurnitureFactory) (chairs : ℕ) (desks : ℕ) : Prop :=
  chairs * factory.carpenterHoursChair + desks * factory.carpenterHoursDesk ≤ factory.maxCarpenterHours ∧
  chairs * factory.painterHoursChair + desks * factory.painterHoursDesk ≤ factory.maxPainterHours

/-- Theorem stating that the maximum profit is 21000 yuan -/
theorem max_profit_is_21000 (factory : FurnitureFactory) 
  (h1 : factory.carpenterHoursChair = 4)
  (h2 : factory.carpenterHoursDesk = 8)
  (h3 : factory.maxCarpenterHours = 8000)
  (h4 : factory.painterHoursChair = 2)
  (h5 : factory.painterHoursDesk = 1)
  (h6 : factory.maxPainterHours = 1300)
  (h7 : factory.profitChair = 15)
  (h8 : factory.profitDesk = 20) :
  (∀ chairs desks, isFeasible factory chairs desks → calculateProfit factory chairs desks ≤ 21000) ∧
  (∃ chairs desks, isFeasible factory chairs desks ∧ calculateProfit factory chairs desks = 21000) :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_21000_l1596_159631


namespace NUMINAMATH_CALUDE_a_mod_4_is_2_or_3_a_not_perfect_square_l1596_159643

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n * a (n + 1) + 1

theorem a_mod_4_is_2_or_3 (n : ℕ) (h : n ≥ 2) : 
  (a n) % 4 = 2 ∨ (a n) % 4 = 3 :=
by sorry

theorem a_not_perfect_square (n : ℕ) (h : n ≥ 2) : 
  ¬ ∃ (k : ℕ), a n = k * k :=
by sorry

end NUMINAMATH_CALUDE_a_mod_4_is_2_or_3_a_not_perfect_square_l1596_159643


namespace NUMINAMATH_CALUDE_fencing_cost_9m_square_l1596_159661

/-- Cost of fencing for each side of a square -/
structure FencingCost where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculate the total cost of fencing a square -/
def totalCost (cost : FencingCost) (sideLength : ℕ) : ℕ :=
  (cost.first + cost.second + cost.third + cost.fourth) * sideLength

/-- The fencing costs for the problem -/
def givenCost : FencingCost :=
  { first := 79
    second := 92
    third := 85
    fourth := 96 }

/-- Theorem: The total cost of fencing the square with side length 9 meters is $3168 -/
theorem fencing_cost_9m_square (cost : FencingCost := givenCost) :
  totalCost cost 9 = 3168 := by
  sorry

#eval totalCost givenCost 9

end NUMINAMATH_CALUDE_fencing_cost_9m_square_l1596_159661


namespace NUMINAMATH_CALUDE_alyona_floor_l1596_159648

/-- Represents a multi-story building with multiple entrances -/
structure Building where
  stories : ℕ
  apartments_per_floor : ℕ
  entrances : ℕ

/-- Calculates the floor number given an apartment number and building structure -/
def floor_number (b : Building) (apartment : ℕ) : ℕ :=
  let apartments_per_entrance := b.stories * b.apartments_per_floor
  let apartments_before_entrance := ((apartment - 1) / apartments_per_entrance) * apartments_per_entrance
  let remaining_apartments := apartment - apartments_before_entrance
  ((remaining_apartments - 1) / b.apartments_per_floor) + 1

/-- Theorem stating that Alyona lives on the 3rd floor -/
theorem alyona_floor :
  ∀ (b : Building),
    b.stories = 9 →
    b.entrances ≥ 10 →
    floor_number b 333 = 3 :=
by sorry

end NUMINAMATH_CALUDE_alyona_floor_l1596_159648


namespace NUMINAMATH_CALUDE_sum_of_ages_l1596_159602

/-- The sum of Mario and Maria's ages is 7 years -/
theorem sum_of_ages : 
  ∀ (mario_age maria_age : ℕ),
  mario_age = 4 →
  mario_age = maria_age + 1 →
  mario_age + maria_age = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1596_159602


namespace NUMINAMATH_CALUDE_tunnel_length_is_1200_l1596_159656

/-- Calculates the length of a tunnel given train specifications and crossing times. -/
def tunnel_length (train_length platform_length : ℝ) 
                  (tunnel_time platform_time : ℝ) : ℝ :=
  3 * (train_length + platform_length) - train_length

/-- Proves that the tunnel length is 1200 meters given the specified conditions. -/
theorem tunnel_length_is_1200 :
  tunnel_length 330 180 45 15 = 1200 := by
  sorry

#eval tunnel_length 330 180 45 15

end NUMINAMATH_CALUDE_tunnel_length_is_1200_l1596_159656


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1596_159635

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 4 * x + 8) * (2 * x + 1) - (2 * x + 1) * (x^2 + 5 * x - 72) + (4 * x - 15) * (2 * x + 1) * (x + 6) =
  12 * x^3 + 22 * x^2 - 12 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1596_159635


namespace NUMINAMATH_CALUDE_rabbit_calories_l1596_159665

/-- Brandon's hunting scenario -/
structure HuntingScenario where
  squirrels_per_hour : ℕ := 6
  rabbits_per_hour : ℕ := 2
  calories_per_squirrel : ℕ := 300
  calorie_difference : ℕ := 200

/-- Calculates the calories per rabbit in Brandon's hunting scenario -/
def calories_per_rabbit (scenario : HuntingScenario) : ℕ :=
  (scenario.squirrels_per_hour * scenario.calories_per_squirrel - scenario.calorie_difference) / scenario.rabbits_per_hour

/-- Theorem stating that each rabbit has 800 calories in Brandon's scenario -/
theorem rabbit_calories (scenario : HuntingScenario) :
  calories_per_rabbit scenario = 800 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_calories_l1596_159665


namespace NUMINAMATH_CALUDE_frank_final_balance_l1596_159664

def frank_money_problem (initial_amount : ℤ) 
                        (game_cost : ℤ) 
                        (keychain_cost : ℤ) 
                        (friend_gift : ℤ) 
                        (allowance : ℤ) 
                        (bus_ticket_cost : ℤ) : Prop :=
  initial_amount = 11 ∧
  game_cost = 3 ∧
  keychain_cost = 2 ∧
  friend_gift = 4 ∧
  allowance = 14 ∧
  bus_ticket_cost = 5 ∧
  initial_amount - game_cost - keychain_cost + friend_gift + allowance - bus_ticket_cost = 19

theorem frank_final_balance :
  ∀ (initial_amount game_cost keychain_cost friend_gift allowance bus_ticket_cost : ℤ),
  frank_money_problem initial_amount game_cost keychain_cost friend_gift allowance bus_ticket_cost :=
by
  sorry

end NUMINAMATH_CALUDE_frank_final_balance_l1596_159664


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1596_159626

theorem complex_number_in_fourth_quadrant (m : ℝ) (h : 1 < m ∧ m < 2) :
  let z : ℂ := Complex.mk (m - 1) (m - 2)
  0 < z.re ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1596_159626


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l1596_159633

theorem sqrt_sum_fractions : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l1596_159633


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l1596_159673

theorem geometric_arithmetic_progression_sum (b q : ℝ) (h1 : b > 0) (h2 : q > 0) :
  let a := b
  let d := (b * q^3 - b) / 3
  (∃ (n : ℕ), q^n = 2) →
  (3 * a + 10 * d = 148 / 9) →
  (b * (q^4 - 1) / (q - 1) = 700 / 27) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l1596_159673


namespace NUMINAMATH_CALUDE_confectioner_pastries_l1596_159641

theorem confectioner_pastries :
  ∀ (P : ℕ) (x : ℕ),
    (P = 28 * (10 + x)) →
    (P = 49 * (4 + x)) →
    P = 392 :=
by
  sorry

end NUMINAMATH_CALUDE_confectioner_pastries_l1596_159641


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1596_159676

-- Define the cycling parameters
def cycling_speed : ℝ := 20
def cycling_time : ℝ := 1

-- Define the walking parameters
def walking_speed : ℝ := 3
def walking_time : ℝ := 2

-- Define the total distance and time
def total_distance : ℝ := cycling_speed * cycling_time + walking_speed * walking_time
def total_time : ℝ := cycling_time + walking_time

-- Theorem statement
theorem average_speed_calculation :
  total_distance / total_time = 26 / 3 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1596_159676


namespace NUMINAMATH_CALUDE_complex_number_equality_l1596_159675

theorem complex_number_equality (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / Complex.I
  (z.re = z.im) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1596_159675


namespace NUMINAMATH_CALUDE_rosie_pie_making_l1596_159669

/-- Represents the number of pies that can be made from a given number of apples -/
def pies_from_apples (apples : ℚ) : ℚ :=
  (2 / 9) * apples

/-- Represents the number of apples left after making pies -/
def apples_left (total_apples : ℚ) (pies_made : ℚ) : ℚ :=
  total_apples - (pies_made * (9 / 2))

theorem rosie_pie_making (total_apples : ℚ) 
  (h1 : total_apples = 36) : 
  pies_from_apples total_apples = 8 ∧ 
  apples_left total_apples (pies_from_apples total_apples) = 0 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pie_making_l1596_159669


namespace NUMINAMATH_CALUDE_final_balloon_count_l1596_159694

def total_balloons (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (tracy_added : ℕ) : ℕ :=
  (brooke_initial + brooke_added) + ((tracy_initial + tracy_added) / 2)

theorem final_balloon_count :
  total_balloons 12 8 6 24 = 35 := by
  sorry

end NUMINAMATH_CALUDE_final_balloon_count_l1596_159694


namespace NUMINAMATH_CALUDE_max_equalization_value_l1596_159636

/-- Represents a 3x3 board with numbers --/
def Board := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if all elements in the board are equal --/
def all_equal (b : Board) : Prop :=
  ∀ i j k l, b i j = b k l

/-- Represents a valid operation on the board --/
inductive Operation
| row (i : Fin 3) (x : ℝ)
| col (j : Fin 3) (x : ℝ)

/-- Applies an operation to the board --/
def apply_operation (b : Board) (op : Operation) : Board :=
  sorry

/-- Checks if a board can be transformed to have all elements equal to m --/
def can_equalize (b : Board) (m : ℕ) : Prop :=
  ∃ (ops : List Operation), all_equal (ops.foldl apply_operation b) ∧
    ∀ i j, (ops.foldl apply_operation b) i j = m

/-- Initial board configuration --/
def initial_board : Board :=
  λ i j => i.val * 3 + j.val + 1

/-- Main theorem: The maximum value of m for which the board can be equalized is 4 --/
theorem max_equalization_value :
  (∀ m : ℕ, m > 4 → ¬ can_equalize initial_board m) ∧
  can_equalize initial_board 4 :=
sorry

end NUMINAMATH_CALUDE_max_equalization_value_l1596_159636


namespace NUMINAMATH_CALUDE_family_ages_l1596_159671

/-- Family ages problem -/
theorem family_ages :
  ∀ (son_age man_age daughter_age wife_age : ℝ),
  (man_age = son_age + 29) →
  (man_age + 2 = 2 * (son_age + 2)) →
  (daughter_age = son_age - 3.5) →
  (wife_age = 1.5 * daughter_age) →
  (son_age = 27 ∧ man_age = 56 ∧ daughter_age = 23.5 ∧ wife_age = 35.25) :=
by
  sorry

#check family_ages

end NUMINAMATH_CALUDE_family_ages_l1596_159671


namespace NUMINAMATH_CALUDE_transformations_correctness_l1596_159653

-- Define the transformations
def transformation_A (a b c : ℝ) : Prop := (c ≠ 0) → (a * c) / (b * c) = a / b

def transformation_B (a b : ℝ) : Prop := (a + b ≠ 0) → (-a - b) / (a + b) = -1

def transformation_C (m n : ℝ) : Prop := 
  (0.2 * m - 0.3 * n ≠ 0) → (0.5 * m + n) / (0.2 * m - 0.3 * n) = (5 * m + 10 * n) / (2 * m - 3 * n)

def transformation_D (x : ℝ) : Prop := (x + 1 ≠ 0) → (2 - x) / (x + 1) = (x - 2) / (1 + x)

-- Theorem stating which transformations are correct and which is incorrect
theorem transformations_correctness :
  (∀ a b c, transformation_A a b c) ∧
  (∀ a b, transformation_B a b) ∧
  (∀ m n, transformation_C m n) ∧
  ¬(∀ x, transformation_D x) := by
  sorry

end NUMINAMATH_CALUDE_transformations_correctness_l1596_159653


namespace NUMINAMATH_CALUDE_remainder_of_2457634_div_8_l1596_159652

theorem remainder_of_2457634_div_8 : 2457634 % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2457634_div_8_l1596_159652


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l1596_159672

theorem popped_kernel_probability (total : ℝ) (white : ℝ) (yellow : ℝ) 
  (white_pop_rate : ℝ) (yellow_pop_rate : ℝ) :
  white / total = 3 / 4 →
  yellow / total = 1 / 4 →
  white_pop_rate = 2 / 5 →
  yellow_pop_rate = 3 / 4 →
  (white * white_pop_rate) / ((white * white_pop_rate) + (yellow * yellow_pop_rate)) = 24 / 39 := by
  sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l1596_159672


namespace NUMINAMATH_CALUDE_sum_of_c_values_l1596_159681

theorem sum_of_c_values : ∃ (S : Finset ℤ),
  (∀ c ∈ S, c ≤ 30 ∧ 
    ∃ x y : ℚ, y = x^2 - 8*x - c ∧ 
    ∃ k : ℤ, (64 + 4*c = k^2)) ∧
  (∀ c : ℤ, c ≤ 30 → 
    (∃ x y : ℚ, y = x^2 - 8*x - c ∧ 
    ∃ k : ℤ, (64 + 4*c = k^2)) → 
    c ∈ S) ∧
  S.sum id = -11 :=
sorry

end NUMINAMATH_CALUDE_sum_of_c_values_l1596_159681


namespace NUMINAMATH_CALUDE_framed_photo_ratio_l1596_159634

/-- Represents the dimensions of a framed photograph -/
structure FramedPhoto where
  original_width : ℝ
  original_height : ℝ
  frame_width : ℝ

/-- Calculates the area of the original photograph -/
def original_area (photo : FramedPhoto) : ℝ :=
  photo.original_width * photo.original_height

/-- Calculates the area of the framed photograph -/
def framed_area (photo : FramedPhoto) : ℝ :=
  (photo.original_width + 2 * photo.frame_width) * (photo.original_height + 6 * photo.frame_width)

/-- Theorem: The ratio of the shorter to the longer dimension of the framed photograph is 1:2 -/
theorem framed_photo_ratio (photo : FramedPhoto) 
  (h1 : photo.original_width = 20)
  (h2 : photo.original_height = 30)
  (h3 : framed_area photo = 2 * original_area photo) :
  (photo.original_width + 2 * photo.frame_width) / (photo.original_height + 6 * photo.frame_width) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_framed_photo_ratio_l1596_159634


namespace NUMINAMATH_CALUDE_justin_bought_two_striped_jerseys_l1596_159696

/-- The number of striped jerseys Justin bought -/
def num_striped_jerseys : ℕ := 2

/-- The cost of each long-sleeved jersey -/
def long_sleeve_cost : ℕ := 15

/-- The number of long-sleeved jerseys Justin bought -/
def num_long_sleeve : ℕ := 4

/-- The cost of each striped jersey before discount -/
def striped_cost : ℕ := 10

/-- The discount applied to each striped jersey after the first one -/
def striped_discount : ℕ := 2

/-- The total amount Justin spent -/
def total_spent : ℕ := 80

/-- Theorem stating that Justin bought 2 striped jerseys given the conditions -/
theorem justin_bought_two_striped_jerseys :
  num_long_sleeve * long_sleeve_cost +
  striped_cost +
  (num_striped_jerseys - 1) * (striped_cost - striped_discount) =
  total_spent :=
sorry

end NUMINAMATH_CALUDE_justin_bought_two_striped_jerseys_l1596_159696


namespace NUMINAMATH_CALUDE_natashas_distance_l1596_159606

/-- The distance to Natasha's destination given her speed and travel time -/
theorem natashas_distance (speed_limit : ℝ) (over_limit : ℝ) (travel_time : ℝ) 
  (h1 : speed_limit = 50)
  (h2 : over_limit = 10)
  (h3 : travel_time = 1) :
  speed_limit + over_limit * travel_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_natashas_distance_l1596_159606


namespace NUMINAMATH_CALUDE_james_sticker_cost_l1596_159624

/-- Calculates James's share of the cost for stickers --/
theorem james_sticker_cost (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ) : 
  packs = 4 → 
  stickers_per_pack = 30 → 
  cost_per_sticker = 1/10 →
  (packs * stickers_per_pack * cost_per_sticker) / 2 = 6 := by
  sorry

#check james_sticker_cost

end NUMINAMATH_CALUDE_james_sticker_cost_l1596_159624


namespace NUMINAMATH_CALUDE_find_m_l1596_159690

theorem find_m (x₁ x₂ m : ℝ) 
  (h1 : x₁^2 - 3*x₁ + m = 0) 
  (h2 : x₂^2 - 3*x₂ + m = 0)
  (h3 : x₁ + x₂ - x₁*x₂ = 1) : 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_find_m_l1596_159690


namespace NUMINAMATH_CALUDE_wheel_distance_l1596_159647

/-- The distance covered by a wheel given its circumference and number of revolutions -/
theorem wheel_distance (circumference : ℝ) (revolutions : ℝ) :
  circumference = 56 →
  revolutions = 3.002729754322111 →
  circumference * revolutions = 168.1528670416402 := by
  sorry

end NUMINAMATH_CALUDE_wheel_distance_l1596_159647


namespace NUMINAMATH_CALUDE_joe_weight_loss_l1596_159607

/-- Represents Joe's weight loss problem --/
theorem joe_weight_loss 
  (initial_weight : ℝ) 
  (months_on_diet : ℝ) 
  (future_weight : ℝ) 
  (months_until_future_weight : ℝ) 
  (h1 : initial_weight = 222)
  (h2 : months_on_diet = 3)
  (h3 : future_weight = 170)
  (h4 : months_until_future_weight = 3.5)
  : ∃ (current_weight : ℝ), 
    current_weight = initial_weight - (initial_weight - future_weight) * (months_on_diet / (months_on_diet + months_until_future_weight))
    ∧ current_weight = 198 :=
by sorry

end NUMINAMATH_CALUDE_joe_weight_loss_l1596_159607


namespace NUMINAMATH_CALUDE_unique_base_for_625_l1596_159638

def is_four_digit (n : ℕ) (b : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

def last_two_digits_odd (n : ℕ) (b : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ d₄ : ℕ, 
    n = d₁ * b^3 + d₂ * b^2 + d₃ * b^1 + d₄ * b^0 ∧
    d₃ % 2 = 1 ∧ d₄ % 2 = 1

theorem unique_base_for_625 :
  ∃! b : ℕ, b > 1 ∧ is_four_digit 625 b ∧ last_two_digits_odd 625 b :=
sorry

end NUMINAMATH_CALUDE_unique_base_for_625_l1596_159638


namespace NUMINAMATH_CALUDE_expansion_terms_count_expansion_terms_count_equals_66_l1596_159617

theorem expansion_terms_count : Nat :=
  let n : Nat := 10  -- power in (a + b + c)^10
  let k : Nat := 3   -- number of variables (a, b, c)
  Nat.choose (n + k - 1) (k - 1)

theorem expansion_terms_count_equals_66 : expansion_terms_count = 66 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_expansion_terms_count_equals_66_l1596_159617


namespace NUMINAMATH_CALUDE_original_number_proof_l1596_159677

theorem original_number_proof (N : ℕ) : 
  (∃ k : ℕ, N - 33 = 87 * k) ∧ 
  (∀ m : ℕ, m < 33 → ¬∃ j : ℕ, N - m = 87 * j) → 
  N = 120 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1596_159677


namespace NUMINAMATH_CALUDE_inequality_proof_l1596_159685

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  ((1 + a * b) / (1 + a)) ^ 2008 + 
  ((1 + b * c) / (1 + b)) ^ 2008 + 
  ((1 + c * d) / (1 + c)) ^ 2008 + 
  ((1 + d * a) / (1 + d)) ^ 2008 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1596_159685


namespace NUMINAMATH_CALUDE_rectangle_triangle_same_area_altitude_l1596_159699

theorem rectangle_triangle_same_area_altitude (h : ℝ) (w : ℝ) : 
  h > 0 →  -- Altitude is positive
  w > 0 →  -- Width is positive
  12 * h = 12 * w →  -- Areas are equal (12h for triangle, 12w for rectangle)
  w = h :=  -- Width of rectangle equals shared altitude
by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_same_area_altitude_l1596_159699


namespace NUMINAMATH_CALUDE_first_group_size_l1596_159604

/-- The number of students in the first group -/
def first_group_count : ℕ := sorry

/-- The number of students in the second group -/
def second_group_count : ℕ := 11

/-- The total number of students in both groups -/
def total_students : ℕ := 31

/-- The average height of students in centimeters -/
def average_height : ℝ := 20

theorem first_group_size :
  (first_group_count : ℝ) * average_height +
  (second_group_count : ℝ) * average_height =
  (total_students : ℝ) * average_height ∧
  first_group_count + second_group_count = total_students →
  first_group_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l1596_159604


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1596_159600

theorem simplify_and_evaluate : 
  let f (x : ℝ) := (2*x + 4) / (x^2 - 6*x + 9) / ((2*x - 1) / (x - 3) - 1)
  f 0 = -2/3 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1596_159600


namespace NUMINAMATH_CALUDE_numeric_methods_students_count_second_year_students_count_l1596_159609

/-- The number of second-year students studying numeric methods -/
def numeric_methods_students : ℕ := 241

/-- The number of second-year students studying automatic control of airborne vehicles -/
def acav_students : ℕ := 423

/-- The number of second-year students studying both numeric methods and ACAV -/
def both_subjects_students : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 663

/-- The proportion of second-year students in the faculty -/
def second_year_proportion : ℚ := 4/5

/-- The total number of second-year students -/
def total_second_year_students : ℕ := 530

theorem numeric_methods_students_count :
  numeric_methods_students + acav_students - both_subjects_students = total_second_year_students :=
by sorry

theorem second_year_students_count :
  total_second_year_students = (total_students : ℚ) * second_year_proportion :=
by sorry

end NUMINAMATH_CALUDE_numeric_methods_students_count_second_year_students_count_l1596_159609


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1596_159630

/-- A hyperbola is defined by its equation and properties --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eqn : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0
  imaginary_axis : b = 1
  asymptote : (x : ℝ) → x / 2 = a / b

/-- The theorem states that a hyperbola with given properties has a specific equation --/
theorem hyperbola_equation (h : Hyperbola) : 
  ∀ x y : ℝ, x^2 / 4 - y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1596_159630


namespace NUMINAMATH_CALUDE_jessica_withdrawal_l1596_159640

theorem jessica_withdrawal (initial_balance : ℝ) (withdrawal : ℝ) : 
  withdrawal = (2 / 5) * initial_balance ∧
  (3 / 5) * initial_balance + (1 / 2) * ((3 / 5) * initial_balance) = 450 →
  withdrawal = 200 := by
sorry

end NUMINAMATH_CALUDE_jessica_withdrawal_l1596_159640


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_77pi_l1596_159686

/-- Represents a triangular pyramid with vertices P, A, B, C -/
structure TriangularPyramid where
  PA : ℝ
  BC : ℝ
  AC : ℝ
  BP : ℝ
  CP : ℝ
  AB : ℝ

/-- The surface area of the circumscribed sphere of a triangular pyramid -/
def circumscribedSphereSurfaceArea (t : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The surface area of the circumscribed sphere of the given triangular pyramid is 77π -/
theorem circumscribed_sphere_surface_area_is_77pi :
  let t : TriangularPyramid := {
    PA := 2 * Real.sqrt 13,
    BC := 2 * Real.sqrt 13,
    AC := Real.sqrt 41,
    BP := Real.sqrt 41,
    CP := Real.sqrt 61,
    AB := Real.sqrt 61
  }
  circumscribedSphereSurfaceArea t = 77 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_77pi_l1596_159686


namespace NUMINAMATH_CALUDE_tank_base_diameter_calculation_l1596_159667

/-- The volume of a cylindrical tank in cubic meters. -/
def tank_volume : ℝ := 1848

/-- The depth of the cylindrical tank in meters. -/
def tank_depth : ℝ := 12.00482999321725

/-- The diameter of the base of the cylindrical tank in meters. -/
def tank_base_diameter : ℝ := 24.838

/-- Theorem stating that the diameter of the base of a cylindrical tank with given volume and depth is approximately equal to the calculated value. -/
theorem tank_base_diameter_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |2 * Real.sqrt (tank_volume / (Real.pi * tank_depth)) - tank_base_diameter| < ε :=
sorry

end NUMINAMATH_CALUDE_tank_base_diameter_calculation_l1596_159667


namespace NUMINAMATH_CALUDE_sin_210_degrees_l1596_159662

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l1596_159662


namespace NUMINAMATH_CALUDE_total_paintable_area_is_1624_l1596_159628

/-- The number of bedrooms in Isabella's house -/
def num_bedrooms : ℕ := 4

/-- The length of each bedroom in feet -/
def bedroom_length : ℕ := 15

/-- The width of each bedroom in feet -/
def bedroom_width : ℕ := 12

/-- The height of each bedroom in feet -/
def bedroom_height : ℕ := 9

/-- The area occupied by doorways and windows in each bedroom in square feet -/
def unpaintable_area : ℕ := 80

/-- The total area of walls to be painted in square feet -/
def total_paintable_area : ℕ := 
  num_bedrooms * (
    2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height) - unpaintable_area
  )

theorem total_paintable_area_is_1624 : total_paintable_area = 1624 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_is_1624_l1596_159628


namespace NUMINAMATH_CALUDE_inequality_solution_l1596_159645

theorem inequality_solution (m : ℝ) (hm : 0 < m ∧ m < 1) :
  {x : ℝ | m * x / (x - 3) > 1} = {x : ℝ | 3 < x ∧ x < 3 / (1 - m)} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1596_159645


namespace NUMINAMATH_CALUDE_zach_cookies_l1596_159693

/-- The number of cookies Zach baked over three days --/
def total_cookies (monday tuesday wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem stating the total number of cookies Zach had after three days --/
theorem zach_cookies : ∃ (monday tuesday wednesday : ℕ),
  monday = 32 ∧
  tuesday = monday / 2 ∧
  wednesday = tuesday * 3 - 4 ∧
  total_cookies monday tuesday wednesday = 92 := by
  sorry

end NUMINAMATH_CALUDE_zach_cookies_l1596_159693


namespace NUMINAMATH_CALUDE_number_equation_solution_l1596_159611

theorem number_equation_solution : ∃ x : ℝ, 
  (0.6667 * x + 1 = 0.75 * x) ∧ 
  (abs (x - 12) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1596_159611


namespace NUMINAMATH_CALUDE_credit_card_balance_proof_l1596_159678

/-- Calculates the new credit card balance after transactions -/
def new_balance (initial_balance groceries_charge towels_return : ℚ) : ℚ :=
  initial_balance + groceries_charge + (groceries_charge / 2) - towels_return

/-- Proves that the new balance is correct given the transactions -/
theorem credit_card_balance_proof :
  new_balance 126 60 45 = 171 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_balance_proof_l1596_159678


namespace NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l1596_159622

theorem triangle_sine_sum_inequality (A B C : Real) : 
  A + B + C = Real.pi → 0 < A → 0 < B → 0 < C →
  Real.sin A + Real.sin B + Real.sin C ≤ (3 / 2) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l1596_159622


namespace NUMINAMATH_CALUDE_string_folding_l1596_159663

theorem string_folding (initial_length : ℝ) (folded_twice : ℕ) : 
  initial_length = 12 ∧ folded_twice = 2 → initial_length / (2^folded_twice) = 3 := by
  sorry

end NUMINAMATH_CALUDE_string_folding_l1596_159663


namespace NUMINAMATH_CALUDE_range_of_g_l1596_159658

theorem range_of_g (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  -Real.pi^2 / 2 ≤ (Real.arccos x)^2 - (Real.arcsin x)^2 ∧ 
  (Real.arccos x)^2 - (Real.arcsin x)^2 ≤ Real.pi^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l1596_159658


namespace NUMINAMATH_CALUDE_two_times_binomial_twelve_choose_three_l1596_159682

theorem two_times_binomial_twelve_choose_three : 2 * (Nat.choose 12 3) = 440 := by
  sorry

end NUMINAMATH_CALUDE_two_times_binomial_twelve_choose_three_l1596_159682


namespace NUMINAMATH_CALUDE_exists_n_plus_S_n_eq_1980_l1596_159642

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of n such that n + S(n) = 1980 -/
theorem exists_n_plus_S_n_eq_1980 : ∃ n : ℕ, n + S n = 1980 := by sorry

end NUMINAMATH_CALUDE_exists_n_plus_S_n_eq_1980_l1596_159642


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1596_159674

def total_marbles : ℕ := 240

theorem yellow_marbles_count (y b : ℕ) 
  (h1 : y + b = total_marbles) 
  (h2 : b = y - 2) : 
  y = 121 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1596_159674


namespace NUMINAMATH_CALUDE_max_M_value_l1596_159632

theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (eq1 : x - 2*y = z - 2*u) (eq2 : 2*y*z = u*x) (h_zy : z ≥ y) :
  ∃ M : ℝ, M > 0 ∧ M ≤ z/y ∧ ∀ N : ℝ, (N > 0 ∧ N ≤ z/y → N ≤ M) ∧ M = 6 + 4*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l1596_159632


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_l1596_159657

/-- The distance between Maxwell's and Brad's homes in kilometers -/
def total_distance : ℝ := 36

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- The distance traveled by Maxwell when they meet -/
def maxwell_distance : ℝ := 12

theorem maxwell_brad_meeting :
  maxwell_distance * brad_speed = (total_distance - maxwell_distance) * maxwell_speed :=
by sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_l1596_159657


namespace NUMINAMATH_CALUDE_min_holiday_days_l1596_159687

/-- Represents a day during the holiday -/
structure Day where
  morning_sunny : Bool
  afternoon_sunny : Bool

/-- Conditions for the holiday weather -/
def valid_holiday (days : List Day) : Prop :=
  let total_days := days.length
  let rainy_days := days.filter (fun d => ¬d.morning_sunny ∨ ¬d.afternoon_sunny)
  let sunny_afternoons := days.filter (fun d => d.afternoon_sunny)
  let sunny_mornings := days.filter (fun d => d.morning_sunny)
  rainy_days.length = 7 ∧
  days.all (fun d => ¬d.afternoon_sunny → d.morning_sunny) ∧
  sunny_afternoons.length = 5 ∧
  sunny_mornings.length = 6

/-- The theorem to be proved -/
theorem min_holiday_days :
  ∃ (days : List Day), valid_holiday days ∧
    ∀ (other_days : List Day), valid_holiday other_days → days.length ≤ other_days.length :=
by
  sorry

end NUMINAMATH_CALUDE_min_holiday_days_l1596_159687


namespace NUMINAMATH_CALUDE_line_equation_solution_l1596_159679

theorem line_equation_solution (a b : ℝ) (h_a : a ≠ 0) :
  (∀ x y : ℝ, y = a * x + b) →
  (4 = a * 0 + b) →
  (0 = a * (-3) + b) →
  (∀ x : ℝ, a * x + b = 0 ↔ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_solution_l1596_159679


namespace NUMINAMATH_CALUDE_fifth_term_value_l1596_159668

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, 2 * a (n + 1) = a n

theorem fifth_term_value (a : ℕ → ℚ) (h : geometric_sequence a) : a 5 = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l1596_159668


namespace NUMINAMATH_CALUDE_circle1_satisfies_conditions_circle2_passes_through_points_l1596_159618

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 10
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0

-- Define the line equation
def line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Theorem for the first circle
theorem circle1_satisfies_conditions :
  (∃ x y : ℝ, line x y ∧ circle1 x y) ∧
  circle1 2 (-3) ∧
  circle1 (-2) (-5) := by sorry

-- Theorem for the second circle
theorem circle2_passes_through_points :
  circle2 1 0 ∧
  circle2 (-1) (-2) ∧
  circle2 3 (-2) := by sorry

end NUMINAMATH_CALUDE_circle1_satisfies_conditions_circle2_passes_through_points_l1596_159618


namespace NUMINAMATH_CALUDE_largest_prime_divisor_exists_l1596_159651

def base_5_number : ℕ := 2031357

theorem largest_prime_divisor_exists :
  ∃ p : ℕ, Prime p ∧ p ∣ base_5_number ∧ ∀ q : ℕ, Prime q → q ∣ base_5_number → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_exists_l1596_159651


namespace NUMINAMATH_CALUDE_exists_zero_sum_subset_l1596_159639

/-- Represents a row in the table -/
def Row (n : ℕ) := Fin n → Int

/-- The table with all possible rows of 1 and -1 -/
def OriginalTable (n : ℕ) : Finset (Row n) :=
  sorry

/-- A function that potentially replaces some elements with zero -/
def Corrupt (n : ℕ) : Row n → Row n :=
  sorry

/-- The corrupted table after replacing some elements with zero -/
def CorruptedTable (n : ℕ) : Finset (Row n) :=
  sorry

/-- Sum of a set of rows -/
def RowSum (n : ℕ) (rows : Finset (Row n)) : Row n :=
  sorry

/-- A row of all zeros -/
def ZeroRow (n : ℕ) : Row n :=
  sorry

/-- The main theorem -/
theorem exists_zero_sum_subset (n : ℕ) :
  ∃ (subset : Finset (Row n)), subset ⊆ CorruptedTable n ∧ RowSum n subset = ZeroRow n :=
sorry

end NUMINAMATH_CALUDE_exists_zero_sum_subset_l1596_159639


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1596_159627

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | -3 ≤ x ∧ x < 3} := by sorry

-- Theorem for A ∩ (∁U B)
theorem intersection_A_complement_B : A ∩ (U \ B) = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l1596_159627


namespace NUMINAMATH_CALUDE_happy_equation_properties_l1596_159603

def happy_number (a b c : ℤ) : ℚ :=
  (4 * a * c - b^2) / (4 * a)

def happy_numbers_to_each_other (a b c p q r : ℤ) : Prop :=
  |r * happy_number a b c - c * happy_number p q r| = 0

theorem happy_equation_properties :
  ∀ (a b c m n p q r : ℤ),
  (a ≠ 0 ∧ p ≠ 0) →
  (∃ (x y : ℤ), a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) →
  (∃ (x y : ℤ), p * x^2 + q * x + r = 0 ∧ p * y^2 + q * y + r = 0 ∧ x ≠ y) →
  (happy_number 1 (-2) (-3) = -4) ∧
  (1 < m ∧ m < 6 ∧ 
   ∃ (x y : ℤ), x^2 - (2*m-1)*x + (m^2-2*m-3) = 0 ∧ 
                y^2 - (2*m-1)*y + (m^2-2*m-3) = 0 ∧ 
                x ≠ y →
   m = 3 ∧ happy_number 1 (-5) 0 = -25/4) ∧
  (∃ (x1 y1 x2 y2 : ℤ),
    x1^2 - m*x1 + (m+1) = 0 ∧ y1^2 - m*y1 + (m+1) = 0 ∧ x1 ≠ y1 ∧
    x2^2 - (n+2)*x2 + 2*n = 0 ∧ y2^2 - (n+2)*y2 + 2*n = 0 ∧ x2 ≠ y2 ∧
    happy_numbers_to_each_other 1 (-m) (m+1) 1 (-(n+2)) (2*n) →
    n = 0 ∨ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_happy_equation_properties_l1596_159603


namespace NUMINAMATH_CALUDE_solution_set_implies_a_greater_than_negative_one_l1596_159623

theorem solution_set_implies_a_greater_than_negative_one (a : ℝ) :
  (∀ x : ℝ, x * (x - a + 1) > a ↔ (x < -1 ∨ x > a)) →
  a > -1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_greater_than_negative_one_l1596_159623


namespace NUMINAMATH_CALUDE_circle_theorem_l1596_159698

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center lies
def centerLine (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the resulting circle
def resultCircle (x y : ℝ) : Prop := (x - 1/2)^2 + (y + 7/2)^2 = 89/2

-- Theorem statement
theorem circle_theorem :
  ∃ (x1 y1 x2 y2 : ℝ),
    -- Intersection points of the two given circles
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    -- The resulting circle passes through these intersection points
    resultCircle x1 y1 ∧ resultCircle x2 y2 ∧
    -- The center of the resulting circle lies on the given line
    centerLine (1/2) (-7/2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_theorem_l1596_159698


namespace NUMINAMATH_CALUDE_exists_integer_function_double_application_square_l1596_159692

theorem exists_integer_function_double_application_square :
  ∃ f : ℤ → ℤ, ∀ n : ℤ, f (f n) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_function_double_application_square_l1596_159692


namespace NUMINAMATH_CALUDE_square_root_pattern_square_root_ten_squared_minus_one_l1596_159660

theorem square_root_pattern (n : ℕ) (hn : n ≥ 3) :
  ∀ m : ℕ, m ≥ 3 → m ≤ 5 →
  Real.sqrt (m^2 - 1) = Real.sqrt (m - 1) * Real.sqrt (m + 1) :=
  sorry

theorem square_root_ten_squared_minus_one :
  Real.sqrt (10^2 - 1) = 3 * Real.sqrt 11 :=
  sorry

end NUMINAMATH_CALUDE_square_root_pattern_square_root_ten_squared_minus_one_l1596_159660


namespace NUMINAMATH_CALUDE_emmalyn_fence_count_l1596_159670

/-- The number of fences Emmalyn painted -/
def number_of_fences : ℕ := 50

/-- The price per meter in dollars -/
def price_per_meter : ℚ := 0.20

/-- The length of each fence in meters -/
def fence_length : ℕ := 500

/-- The total earnings in dollars -/
def total_earnings : ℕ := 5000

/-- Theorem stating that the number of fences Emmalyn painted is correct -/
theorem emmalyn_fence_count :
  number_of_fences = total_earnings / (price_per_meter * fence_length) := by
  sorry

end NUMINAMATH_CALUDE_emmalyn_fence_count_l1596_159670


namespace NUMINAMATH_CALUDE_simplify_expression_l1596_159697

theorem simplify_expression : 
  Real.sqrt 6 * 6^(1/2) + 18 / 3 * 4 - (2 + 2)^(5/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1596_159697


namespace NUMINAMATH_CALUDE_shopping_money_calculation_l1596_159683

theorem shopping_money_calculation (initial_amount : ℝ) : 
  (0.7 * initial_amount = 350) → initial_amount = 500 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_calculation_l1596_159683


namespace NUMINAMATH_CALUDE_range_of_a_l1596_159601

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1596_159601


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l1596_159646

-- Define the sets M and N
def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | x > 1}

-- Theorem statement
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ 
  (∃ a : ℝ, a ∈ M ∧ a ∉ N) :=
by sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l1596_159646


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1596_159688

theorem simplify_and_evaluate (a : ℤ) : 
  2 * (4 * a ^ 2 - a) - (3 * a ^ 2 - 2 * a + 5) = 40 ↔ a = -3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1596_159688


namespace NUMINAMATH_CALUDE_min_value_expression_l1596_159650

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 9) :
  (a^2 + b^2 + c^2)/(a + b + c) + (b^2 + c^2)/(b + c) + (c^2 + a^2)/(c + a) + (a^2 + b^2)/(a + b) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1596_159650


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l1596_159691

theorem division_multiplication_problem : (180 / 6) / 3 * 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l1596_159691


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1596_159655

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1596_159655


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l1596_159608

theorem sqrt_3_irrational (numbers : Set ℝ) (h1 : numbers = {-1, 0, (1/2 : ℝ), Real.sqrt 3}) :
  ∃ x ∈ numbers, Irrational x ∧ ∀ y ∈ numbers, y ≠ x → ¬ Irrational y :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l1596_159608


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1596_159620

def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ
  | n => a₁ * q ^ (n - 1)

theorem geometric_sequence_sum (a₁ q : ℝ) :
  ∃ (a₁ q : ℝ),
    (geometric_sequence a₁ q 2 + geometric_sequence a₁ q 4 = 20) ∧
    (geometric_sequence a₁ q 3 + geometric_sequence a₁ q 5 = 40) →
    (geometric_sequence a₁ q 5 + geometric_sequence a₁ q 7 = 160) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1596_159620


namespace NUMINAMATH_CALUDE_chandra_reading_pages_l1596_159629

/-- Represents the number of pages in the book -/
def total_pages : ℕ := 900

/-- Represents Chandra's reading speed in seconds per page -/
def chandra_speed : ℕ := 30

/-- Represents Daniel's reading speed in seconds per page -/
def daniel_speed : ℕ := 60

/-- Calculates the number of pages Chandra should read -/
def chandra_pages : ℕ := total_pages * daniel_speed / (chandra_speed + daniel_speed)

theorem chandra_reading_pages :
  chandra_pages = 600 ∧
  chandra_pages * chandra_speed = (total_pages - chandra_pages) * daniel_speed :=
by sorry

end NUMINAMATH_CALUDE_chandra_reading_pages_l1596_159629


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1596_159637

/-- A rectangular solid with prime edge lengths and volume 429 has surface area 430. -/
theorem rectangular_solid_surface_area :
  ∀ l w h : ℕ,
  Prime l → Prime w → Prime h →
  l * w * h = 429 →
  2 * (l * w + w * h + h * l) = 430 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1596_159637


namespace NUMINAMATH_CALUDE_cubic_equation_coefficient_sum_of_squares_l1596_159689

theorem cubic_equation_coefficient_sum_of_squares :
  ∀ (p q r s t u : ℤ),
  (∀ x, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_coefficient_sum_of_squares_l1596_159689


namespace NUMINAMATH_CALUDE_knocks_to_knicks_conversion_l1596_159649

/-- Conversion rate between knicks and knacks -/
def knicks_to_knacks : ℚ := 3 / 8

/-- Conversion rate between knacks and knocks -/
def knacks_to_knocks : ℚ := 6 / 5

/-- The number of knocks we want to convert -/
def target_knocks : ℚ := 30

theorem knocks_to_knicks_conversion :
  target_knocks * knacks_to_knocks⁻¹ * knicks_to_knacks⁻¹ = 200 / 3 :=
sorry

end NUMINAMATH_CALUDE_knocks_to_knicks_conversion_l1596_159649


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_21_over_5_l1596_159625

theorem greatest_integer_less_than_negative_21_over_5 :
  Int.floor (-21 / 5 : ℚ) = -5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_21_over_5_l1596_159625


namespace NUMINAMATH_CALUDE_inequality_chain_l1596_159605

theorem inequality_chain (m n : ℝ) 
  (hm : m < 0) 
  (hn : n > 0) 
  (hmn : m + n < 0) : 
  m < -n ∧ -n < n ∧ n < -m :=
by sorry

end NUMINAMATH_CALUDE_inequality_chain_l1596_159605


namespace NUMINAMATH_CALUDE_misha_money_total_l1596_159619

theorem misha_money_total (initial_money earned_money : ℕ) : 
  initial_money = 34 → earned_money = 13 → initial_money + earned_money = 47 := by
  sorry

end NUMINAMATH_CALUDE_misha_money_total_l1596_159619


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l1596_159666

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l1596_159666


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1596_159613

/-- A geometric sequence with common ratio 2 and specific sum condition -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The sum of the 3rd, 4th, and 5th terms equals 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
    a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1596_159613


namespace NUMINAMATH_CALUDE_chapter_page_difference_l1596_159610

theorem chapter_page_difference (first_chapter_pages second_chapter_pages : ℕ) 
  (h1 : first_chapter_pages = 48) 
  (h2 : second_chapter_pages = 11) : 
  first_chapter_pages - second_chapter_pages = 37 := by
  sorry

end NUMINAMATH_CALUDE_chapter_page_difference_l1596_159610


namespace NUMINAMATH_CALUDE_suzanna_ride_l1596_159695

/-- Calculates the distance traveled given a constant rate and time -/
def distanceTraveled (rate : ℚ) (time : ℚ) : ℚ :=
  rate * time

theorem suzanna_ride : 
  let rate : ℚ := 1.5 / 4  -- miles per minute
  let time : ℚ := 40       -- minutes
  distanceTraveled rate time = 15 := by
sorry

#eval (1.5 / 4) * 40  -- To verify the result

end NUMINAMATH_CALUDE_suzanna_ride_l1596_159695


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l1596_159684

theorem pizza_payment_difference :
  -- Define the total number of slices
  let total_slices : ℕ := 12
  -- Define the cost of the plain pizza
  let plain_cost : ℚ := 12
  -- Define the additional cost for mushrooms
  let mushroom_cost : ℚ := 3
  -- Define the number of slices with mushrooms (one-third of the pizza)
  let mushroom_slices : ℕ := total_slices / 3
  -- Define the number of slices Laura ate
  let laura_slices : ℕ := mushroom_slices + 2
  -- Define the number of slices Jessica ate
  let jessica_slices : ℕ := total_slices - laura_slices
  -- Calculate the total cost of the pizza
  let total_cost : ℚ := plain_cost + mushroom_cost
  -- Calculate the cost per slice
  let cost_per_slice : ℚ := total_cost / total_slices
  -- Calculate Laura's payment
  let laura_payment : ℚ := laura_slices * cost_per_slice
  -- Calculate Jessica's payment (only plain slices)
  let jessica_payment : ℚ := jessica_slices * (plain_cost / total_slices)
  -- The difference in payment
  laura_payment - jessica_payment = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l1596_159684


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1596_159654

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f y < f x

-- State the theorem
theorem solution_set_of_inequality 
  (h_even : is_even f) 
  (h_decreasing : is_decreasing_on_nonneg f) :
  {x : ℝ | f (2*x + 5) > f (x^2 + 2)} = {x : ℝ | x < -1 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1596_159654


namespace NUMINAMATH_CALUDE_jessica_quarters_l1596_159659

/-- Calculates the number of quarters Jessica has after her sister borrows some. -/
def quarters_remaining (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that if Jessica had 8 quarters initially and her sister borrowed 3,
    then Jessica now has 5 quarters. -/
theorem jessica_quarters :
  quarters_remaining 8 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_quarters_l1596_159659


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1596_159615

theorem sum_of_numbers : 2 * 2143 + 4321 + 3214 + 1432 = 13523 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1596_159615


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l1596_159614

theorem isosceles_right_triangle_ratio (a c : ℝ) (h1 : a > 0) (h2 : c > 0) : 
  (a^2 + a^2 = c^2) → (2 * a / c = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l1596_159614


namespace NUMINAMATH_CALUDE_even_function_implies_a_plus_minus_one_l1596_159621

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem even_function_implies_a_plus_minus_one (a : ℝ) :
  EvenFunction (fun x => x^2 + (a^2 - 1)*x + (a - 1)) →
  a = 1 ∨ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_plus_minus_one_l1596_159621


namespace NUMINAMATH_CALUDE_is_circle_center_l1596_159644

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 12*y + 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -6)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center : 
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l1596_159644
