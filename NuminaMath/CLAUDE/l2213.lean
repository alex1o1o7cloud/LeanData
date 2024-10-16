import Mathlib

namespace NUMINAMATH_CALUDE_function_identity_l2213_221304

theorem function_identity (f g h : ℕ → ℕ) 
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2213_221304


namespace NUMINAMATH_CALUDE_cattle_transport_time_l2213_221330

/-- Calculates the total driving time to transport cattle to higher ground -/
def total_driving_time (total_cattle : ℕ) (distance : ℕ) (truck_capacity : ℕ) (speed : ℕ) : ℕ :=
  let num_trips := total_cattle / truck_capacity
  let round_trip_time := 2 * (distance / speed)
  num_trips * round_trip_time

/-- Theorem stating that under given conditions, the total driving time is 40 hours -/
theorem cattle_transport_time : total_driving_time 400 60 20 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cattle_transport_time_l2213_221330


namespace NUMINAMATH_CALUDE_selling_price_with_equal_loss_l2213_221325

/-- Given an article with cost price 59 and selling price 66 resulting in a profit of 7,
    prove that the selling price resulting in the same loss as the profit is 52. -/
theorem selling_price_with_equal_loss (cost_price selling_price_profit : ℕ) 
  (h1 : cost_price = 59)
  (h2 : selling_price_profit = 66)
  (h3 : selling_price_profit - cost_price = 7) : 
  ∃ (selling_price_loss : ℕ), 
    selling_price_loss = 52 ∧ 
    cost_price - selling_price_loss = selling_price_profit - cost_price :=
by sorry

end NUMINAMATH_CALUDE_selling_price_with_equal_loss_l2213_221325


namespace NUMINAMATH_CALUDE_sport_to_standard_ratio_l2213_221388

/-- The ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
| 0 => 1
| 1 => 12
| 2 => 30

/-- The ratio of flavoring to corn syrup in the sport formulation is three times that of standard -/
def sport_ratio_multiplier : ℚ := 3

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- The amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 105

/-- The ratio of flavoring to water in the sport formulation compared to the standard formulation -/
theorem sport_to_standard_ratio : 
  (sport_corn_syrup / sport_ratio_multiplier / sport_water) / 
  (standard_ratio 0 / standard_ratio 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sport_to_standard_ratio_l2213_221388


namespace NUMINAMATH_CALUDE_button_to_magnet_ratio_l2213_221361

/-- Represents the number of earrings in a set -/
def earrings_per_set : ℕ := 2

/-- Represents the number of sets Rebecca wants to make -/
def sets : ℕ := 4

/-- Represents the total number of gemstones needed -/
def total_gemstones : ℕ := 24

/-- Represents the number of magnets used in each earring -/
def magnets_per_earring : ℕ := 2

/-- Represents the ratio of gemstones to buttons -/
def gemstone_to_button_ratio : ℕ := 3

/-- Theorem stating the ratio of buttons to magnets for each earring -/
theorem button_to_magnet_ratio :
  let total_earrings := sets * earrings_per_set
  let total_buttons := total_gemstones / gemstone_to_button_ratio
  let buttons_per_earring := total_buttons / total_earrings
  (buttons_per_earring : ℚ) / magnets_per_earring = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_button_to_magnet_ratio_l2213_221361


namespace NUMINAMATH_CALUDE_water_per_pig_l2213_221349

-- Define the given conditions
def pump_rate : ℚ := 3
def pumping_time : ℚ := 25
def corn_rows : ℕ := 4
def corn_plants_per_row : ℕ := 15
def water_per_corn_plant : ℚ := 1/2
def num_pigs : ℕ := 10
def num_ducks : ℕ := 20
def water_per_duck : ℚ := 1/4

-- Theorem to prove
theorem water_per_pig : 
  (pump_rate * pumping_time - 
   (corn_rows * corn_plants_per_row : ℚ) * water_per_corn_plant - 
   (num_ducks : ℚ) * water_per_duck) / (num_pigs : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_per_pig_l2213_221349


namespace NUMINAMATH_CALUDE_negation_equivalence_l2213_221336

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2213_221336


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l2213_221387

theorem abs_inequality_solution_set (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l2213_221387


namespace NUMINAMATH_CALUDE_bruce_shopping_theorem_l2213_221394

/-- Calculates the remaining money after Bruce's shopping trip. -/
def remaining_money (initial_amount shirt_price num_shirts pants_price : ℕ) : ℕ :=
  initial_amount - (shirt_price * num_shirts + pants_price)

/-- Theorem stating that Bruce has $20 left after his shopping trip. -/
theorem bruce_shopping_theorem :
  remaining_money 71 5 5 26 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bruce_shopping_theorem_l2213_221394


namespace NUMINAMATH_CALUDE_smallest_n_squared_existence_of_solution_smallest_n_is_11_l2213_221302

theorem smallest_n_squared (n : ℕ+) : 
  (∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11) →
  n ≥ 11 :=
by sorry

theorem existence_of_solution : 
  ∃ (x y z : ℕ+), 11^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11 :=
by sorry

theorem smallest_n_is_11 : 
  (∃ (n : ℕ+), ∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11) ∧
  (∀ (m : ℕ+), (∃ (x y z : ℕ+), m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 11) → m ≥ 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_squared_existence_of_solution_smallest_n_is_11_l2213_221302


namespace NUMINAMATH_CALUDE_intersection_when_m_zero_m_range_for_necessary_condition_l2213_221366

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define predicates p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

-- Theorem for part (I)
theorem intersection_when_m_zero : 
  A ∩ B 0 = {x | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part (II)
theorem m_range_for_necessary_condition :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) ↔ m ≥ 4 ∨ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_zero_m_range_for_necessary_condition_l2213_221366


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_geq_one_l2213_221376

-- Define sets A and B
def A : Set ℝ := {x | 3 * x + 1 < 4}
def B (a : ℝ) : Set ℝ := {x | x - a < 0}

-- State the theorem
theorem intersection_equality_implies_a_geq_one (a : ℝ) :
  A ∩ B a = A → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_geq_one_l2213_221376


namespace NUMINAMATH_CALUDE_diagonal_game_winner_l2213_221398

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the outcome of the game -/
inductive Outcome
| FirstPlayerWins
| SecondPlayerWins

/-- The number of diagonals in a polygon with s sides -/
def num_diagonals (s : ℕ) : ℕ := s * (s - 3) / 2

/-- The winner of the diagonal drawing game in a (2n+1)-gon -/
def winner (n : ℕ) : Outcome :=
  if n % 2 = 0 then Outcome.FirstPlayerWins else Outcome.SecondPlayerWins

/-- The main theorem about the winner of the diagonal drawing game -/
theorem diagonal_game_winner (n : ℕ) (h : n > 1) :
  winner n = (if n % 2 = 0 then Outcome.FirstPlayerWins else Outcome.SecondPlayerWins) :=
sorry

end NUMINAMATH_CALUDE_diagonal_game_winner_l2213_221398


namespace NUMINAMATH_CALUDE_puzzle_solution_l2213_221372

/-- Represents the possible values in a cell of the grid -/
inductive CellValue
  | Two
  | Zero
  | One
  | Five
  | Empty

/-- Represents a 5x6 grid -/
def Grid := Matrix (Fin 5) (Fin 6) CellValue

/-- Checks if a given grid satisfies the puzzle constraints -/
def is_valid_grid (g : Grid) : Prop :=
  -- Each row contains each digit exactly once
  (∀ i, ∃! j, g i j = CellValue.Two) ∧
  (∀ i, ∃! j, g i j = CellValue.Zero) ∧
  (∀ i, ∃! j, g i j = CellValue.One) ∧
  (∀ i, ∃! j, g i j = CellValue.Five) ∧
  -- Each column contains each digit exactly once
  (∀ j, ∃! i, g i j = CellValue.Two) ∧
  (∀ j, ∃! i, g i j = CellValue.Zero) ∧
  (∀ j, ∃! i, g i j = CellValue.One) ∧
  (∀ j, ∃! i, g i j = CellValue.Five) ∧
  -- Same digits are not adjacent diagonally
  (∀ i j, i < 4 → j < 5 → g i j ≠ g (i+1) (j+1)) ∧
  (∀ i j, i < 4 → j > 0 → g i j ≠ g (i+1) (j-1))

/-- The theorem stating the solution to the puzzle -/
theorem puzzle_solution (g : Grid) (h : is_valid_grid g) :
  g 4 0 = CellValue.One ∧
  g 4 1 = CellValue.Five ∧
  g 4 2 = CellValue.Empty ∧
  g 4 3 = CellValue.Empty ∧
  g 4 4 = CellValue.Two :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2213_221372


namespace NUMINAMATH_CALUDE_temporary_employee_percentage_l2213_221335

theorem temporary_employee_percentage 
  (total_workers : ℕ) 
  (technicians : ℕ) 
  (non_technicians : ℕ) 
  (permanent_technicians : ℕ) 
  (permanent_non_technicians : ℕ) 
  (h1 : technicians = total_workers / 2) 
  (h2 : non_technicians = total_workers / 2) 
  (h3 : permanent_technicians = technicians / 2) 
  (h4 : permanent_non_technicians = non_technicians / 2) :
  (total_workers - (permanent_technicians + permanent_non_technicians)) * 100 / total_workers = 50 := by
  sorry

end NUMINAMATH_CALUDE_temporary_employee_percentage_l2213_221335


namespace NUMINAMATH_CALUDE_h_satisfies_condition_l2213_221351

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := 4 * x - 5

def h (x : ℝ) : ℝ := 2 * x - 4

theorem h_satisfies_condition : ∀ x : ℝ, f (h x) = g x := by
  sorry

end NUMINAMATH_CALUDE_h_satisfies_condition_l2213_221351


namespace NUMINAMATH_CALUDE_certain_to_select_genuine_l2213_221389

/-- A set of products with genuine and defective items -/
structure ProductSet where
  total : ℕ
  genuine : ℕ
  defective : ℕ
  h1 : genuine + defective = total

/-- The number of products to be selected -/
def selection_size : ℕ := 3

/-- The specific product set in the problem -/
def problem_set : ProductSet where
  total := 12
  genuine := 10
  defective := 2
  h1 := by rfl

/-- The probability of selecting at least one genuine product -/
def prob_at_least_one_genuine (ps : ProductSet) : ℚ :=
  1 - (Nat.choose ps.defective selection_size : ℚ) / (Nat.choose ps.total selection_size : ℚ)

theorem certain_to_select_genuine :
  prob_at_least_one_genuine problem_set = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_to_select_genuine_l2213_221389


namespace NUMINAMATH_CALUDE_exponent_division_l2213_221360

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2213_221360


namespace NUMINAMATH_CALUDE_shoe_selection_probability_l2213_221324

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 5

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selected_shoes : ℕ := 2

/-- The number of ways to select 2 shoes of the same color -/
def same_color_selections : ℕ := num_pairs

/-- The total number of ways to select 2 shoes from 10 shoes -/
def total_selections : ℕ := Nat.choose total_shoes selected_shoes

theorem shoe_selection_probability :
  same_color_selections / total_selections = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_shoe_selection_probability_l2213_221324


namespace NUMINAMATH_CALUDE_min_value_vector_expr_l2213_221380

/-- Given plane vectors a, b, and c satisfying certain conditions, 
    the minimum value of a specific vector expression is 1/2. -/
theorem min_value_vector_expr 
  (a b c : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖c‖ = 2) 
  (h4 : a + b + c = (0, 0)) :
  ∃ (min : ℝ), min = 1/2 ∧ 
  ∀ (x y : ℝ), 0 ≤ x → x ≤ 1/2 → 1/2 ≤ y → y ≤ 1 →
  ‖x • (a - c) + y • (b - c) + c‖ ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_vector_expr_l2213_221380


namespace NUMINAMATH_CALUDE_problem_solution_l2213_221339

noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x) - 2 * x

noncomputable def g (b : ℝ) (x : ℝ) := f (2 * x) - 4 * b * f x

theorem problem_solution :
  (∀ x : ℝ, (deriv f) x ≥ 0) ∧
  (∃ b_max : ℝ, b_max = 2 ∧ ∀ b : ℝ, (∀ x : ℝ, x > 0 → g b x > 0) → b ≤ b_max) ∧
  (0.693 < Real.log 2 ∧ Real.log 2 < 0.694) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2213_221339


namespace NUMINAMATH_CALUDE_vector_projection_l2213_221350

theorem vector_projection (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (-2, 1)) :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj = -4 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l2213_221350


namespace NUMINAMATH_CALUDE_range_of_m_l2213_221319

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - m*x + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + m > 0

-- Define the main theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m) → 1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2213_221319


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2213_221377

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + m = 0) ↔ m = 1/9 ∧ m > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l2213_221377


namespace NUMINAMATH_CALUDE_nathan_added_half_blankets_l2213_221352

/-- The fraction of blankets Nathan added to his bed -/
def blanket_fraction (total_blankets : ℕ) (temp_per_blanket : ℕ) (total_temp_increase : ℕ) : ℚ :=
  (total_temp_increase / temp_per_blanket : ℚ) / total_blankets

/-- Theorem stating that Nathan added 1/2 of his blankets -/
theorem nathan_added_half_blankets :
  blanket_fraction 14 3 21 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_nathan_added_half_blankets_l2213_221352


namespace NUMINAMATH_CALUDE_remainder_4123_div_32_l2213_221375

theorem remainder_4123_div_32 : 4123 % 32 = 27 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4123_div_32_l2213_221375


namespace NUMINAMATH_CALUDE_johns_age_relation_l2213_221367

theorem johns_age_relation :
  ∀ (john_age brother_age : ℕ) (x : ℚ),
    john_age = (x * brother_age : ℚ).floor - 4 →
    john_age + brother_age = 10 →
    brother_age = 8 →
    (john_age : ℚ) + 4 = 3/4 * brother_age :=
by
  sorry

end NUMINAMATH_CALUDE_johns_age_relation_l2213_221367


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l2213_221326

theorem quadratic_form_equivalence (d : ℕ) (h : d > 0) (h_div : d ∣ (5 + 2022^2022)) :
  (∃ x y : ℤ, d = 2*x^2 + 2*x*y + 3*y^2) ↔ (d % 20 = 3 ∨ d % 20 = 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l2213_221326


namespace NUMINAMATH_CALUDE_solve_equation_l2213_221348

theorem solve_equation (x : ℝ) (h : x - 2*x + 3*x - 4*x = 120) : x = -60 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2213_221348


namespace NUMINAMATH_CALUDE_life_insurance_amount_l2213_221333

/-- Calculates the life insurance amount given Bobby's salary and deductions --/
theorem life_insurance_amount
  (weekly_salary : ℝ)
  (federal_tax_rate : ℝ)
  (state_tax_rate : ℝ)
  (health_insurance : ℝ)
  (parking_fee : ℝ)
  (final_amount : ℝ)
  (h1 : weekly_salary = 450)
  (h2 : federal_tax_rate = 1/3)
  (h3 : state_tax_rate = 0.08)
  (h4 : health_insurance = 50)
  (h5 : parking_fee = 10)
  (h6 : final_amount = 184) :
  weekly_salary - (weekly_salary * federal_tax_rate) - (weekly_salary * state_tax_rate) - health_insurance - parking_fee - final_amount = 20 := by
  sorry

#check life_insurance_amount

end NUMINAMATH_CALUDE_life_insurance_amount_l2213_221333


namespace NUMINAMATH_CALUDE_three_digit_number_relation_l2213_221384

theorem three_digit_number_relation :
  ∀ a b c : ℕ,
  (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) →  -- three-digit number condition
  (100 * a + 10 * b + c = 56 * c) →                               -- 56 times last digit condition
  (100 * a + 10 * b + c = 112 * a) :=                             -- 112 times first digit (to prove)
by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_relation_l2213_221384


namespace NUMINAMATH_CALUDE_w_squared_value_l2213_221323

theorem w_squared_value (w : ℝ) (h : 2 * (w + 15)^2 = (4 * w + 9) * (3 * w + 6)) :
  w^2 = (9 + Real.sqrt 15921) / 20 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l2213_221323


namespace NUMINAMATH_CALUDE_transformed_point_sum_l2213_221315

/-- Given a function g : ℝ → ℝ such that g(8) = 5, 
    prove that (8/3, 14/9) is on the graph of 3y = g(3x)/3 + 3 
    and that the sum of its coordinates is 38/9 -/
theorem transformed_point_sum (g : ℝ → ℝ) (h : g 8 = 5) : 
  let f : ℝ → ℝ := λ x => (g (3 * x) / 3 + 3) / 3
  f (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
sorry

end NUMINAMATH_CALUDE_transformed_point_sum_l2213_221315


namespace NUMINAMATH_CALUDE_marion_additional_points_l2213_221306

/-- Represents the additional points Marion got on the exam -/
def additional_points (total_items : ℕ) (ella_incorrect : ℕ) (marion_score : ℕ) : ℕ :=
  marion_score - (total_items - ella_incorrect) / 2

/-- Proves that Marion got 6 additional points given the exam conditions -/
theorem marion_additional_points :
  additional_points 40 4 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_marion_additional_points_l2213_221306


namespace NUMINAMATH_CALUDE_conference_handshakes_l2213_221342

/-- Calculates the maximum number of handshakes in a conference with given constraints -/
def max_handshakes (total : ℕ) (committee : ℕ) (red_badges : ℕ) : ℕ :=
  let participants := total - committee - red_badges
  participants * (participants - 1) / 2

/-- Theorem stating the maximum number of handshakes for the given conference -/
theorem conference_handshakes :
  max_handshakes 50 10 5 = 595 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2213_221342


namespace NUMINAMATH_CALUDE_exists_special_sequence_l2213_221301

/-- A sequence of natural numbers -/
def IncreasingSequence := ℕ → ℕ

/-- Property that the sequence is strictly increasing -/
def IsStrictlyIncreasing (a : IncreasingSequence) : Prop :=
  a 0 = 0 ∧ ∀ n : ℕ, a (n + 1) > a n

/-- Property that every natural number is the sum of two sequence terms -/
def HasAllSums (a : IncreasingSequence) : Prop :=
  ∀ k : ℕ, ∃ i j : ℕ, k = a i + a j

/-- Property that each term is greater than n²/16 -/
def SatisfiesLowerBound (a : IncreasingSequence) : Prop :=
  ∀ n : ℕ, n > 0 → a n > (n^2 : ℚ) / 16

/-- The main theorem stating the existence of a sequence satisfying all conditions -/
theorem exists_special_sequence :
  ∃ a : IncreasingSequence, 
    IsStrictlyIncreasing a ∧ 
    HasAllSums a ∧ 
    SatisfiesLowerBound a := by
  sorry

end NUMINAMATH_CALUDE_exists_special_sequence_l2213_221301


namespace NUMINAMATH_CALUDE_distinct_collections_biology_l2213_221340

-- Define the set of letters in BIOLOGY
def biology : Finset Char := {'B', 'I', 'O', 'L', 'G', 'Y'}

-- Define the set of vowels in BIOLOGY
def vowels : Finset Char := {'I', 'O'}

-- Define the set of consonants in BIOLOGY
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

-- Define the number of vowels to be selected
def num_vowels : ℕ := 2

-- Define the number of consonants to be selected
def num_consonants : ℕ := 4

-- Define a function to count distinct collections
def count_distinct_collections : ℕ := sorry

-- Theorem statement
theorem distinct_collections_biology :
  count_distinct_collections = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_biology_l2213_221340


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2213_221357

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- The probability that two randomly chosen diagonals of a regular nonagon intersect inside the nonagon -/
def intersectionProbability (n : RegularNonagon) : ℚ :=
  6 / 13

/-- Theorem: The probability that two randomly chosen diagonals of a regular nonagon 
    intersect inside the nonagon is 6/13 -/
theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) : 
  intersectionProbability n = 6 / 13 := by
  sorry


end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2213_221357


namespace NUMINAMATH_CALUDE_bell_interval_problem_l2213_221379

/-- Represents the intervals of the four bells in seconds -/
structure BellIntervals where
  bell1 : ℕ
  bell2 : ℕ
  bell3 : ℕ
  bell4 : ℕ

/-- Checks if the given intervals result in the bells tolling together after the specified time -/
def tollTogether (intervals : BellIntervals) (time : ℕ) : Prop :=
  time % intervals.bell1 = 0 ∧
  time % intervals.bell2 = 0 ∧
  time % intervals.bell3 = 0 ∧
  time % intervals.bell4 = 0

/-- The main theorem to prove -/
theorem bell_interval_problem (intervals : BellIntervals) :
  intervals.bell1 = 9 →
  intervals.bell3 = 14 →
  intervals.bell4 = 18 →
  tollTogether intervals 630 →
  intervals.bell2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bell_interval_problem_l2213_221379


namespace NUMINAMATH_CALUDE_committee_probability_l2213_221382

/-- The probability of selecting a committee with at least one boy and one girl -/
theorem committee_probability (total : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) : 
  total = 30 →
  boys = 12 →
  girls = 18 →
  committee_size = 6 →
  (1 : ℚ) - (Nat.choose boys committee_size + Nat.choose girls committee_size : ℚ) / 
    (Nat.choose total committee_size : ℚ) = 574287 / 593775 := by
  sorry

#check committee_probability

end NUMINAMATH_CALUDE_committee_probability_l2213_221382


namespace NUMINAMATH_CALUDE_rosie_pies_l2213_221396

def apple_pies_per_batch : ℕ := 2
def apples_per_batch : ℕ := 9
def pear_pies_per_batch : ℕ := 3
def pears_per_batch : ℕ := 15

def available_apples : ℕ := 27
def available_pears : ℕ := 30

def total_pies : ℕ := 
  (available_apples / apples_per_batch) * apple_pies_per_batch +
  (available_pears / pears_per_batch) * pear_pies_per_batch

theorem rosie_pies : total_pies = 12 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_l2213_221396


namespace NUMINAMATH_CALUDE_half_month_days_l2213_221334

/-- Proves that given a 30-day month with specified mean profits, 
    the number of days in each half of the month is 15. -/
theorem half_month_days (total_days : ℕ) (mean_profit : ℚ) 
    (first_half_mean : ℚ) (second_half_mean : ℚ) : 
    total_days = 30 ∧ 
    mean_profit = 350 ∧ 
    first_half_mean = 225 ∧ 
    second_half_mean = 475 → 
    ∃ (first_half_days second_half_days : ℕ), 
      first_half_days = 15 ∧ 
      second_half_days = 15 ∧ 
      first_half_days + second_half_days = total_days ∧
      (first_half_mean * first_half_days + second_half_mean * second_half_days) / total_days = mean_profit :=
by sorry

end NUMINAMATH_CALUDE_half_month_days_l2213_221334


namespace NUMINAMATH_CALUDE_robot_price_ratio_l2213_221341

/-- The ratio of the price Tom should pay to the original price of the robot -/
theorem robot_price_ratio (original_price tom_price : ℚ) 
  (h1 : original_price = 3)
  (h2 : tom_price = 9) :
  tom_price / original_price = 3 := by
sorry

end NUMINAMATH_CALUDE_robot_price_ratio_l2213_221341


namespace NUMINAMATH_CALUDE_zoo_escape_zoo_escape_proof_l2213_221314

theorem zoo_escape (lions : ℕ) (recovery_time : ℕ) (total_time : ℕ) : ℕ :=
  let rhinos := (total_time / recovery_time) - lions
  rhinos

theorem zoo_escape_proof :
  zoo_escape 3 2 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_escape_zoo_escape_proof_l2213_221314


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_700_l2213_221370

theorem greatest_multiple_of_5_and_6_less_than_700 : 
  ∃ n : ℕ, n = 690 ∧ 
  (∀ m : ℕ, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 700 → m ≤ n) ∧
  n % 5 = 0 ∧ n % 6 = 0 ∧ n < 700 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_700_l2213_221370


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l2213_221312

theorem sum_six_consecutive_integers (m : ℤ) : 
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) = 6 * m + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l2213_221312


namespace NUMINAMATH_CALUDE_polar_to_cartesian_intersecting_lines_l2213_221369

/-- The polar coordinate equation ρ(cos²θ - sin²θ) = 0 represents two intersecting lines -/
theorem polar_to_cartesian_intersecting_lines :
  ∃ (x y : ℝ → ℝ), 
    (∀ θ : ℝ, x θ^2 = y θ^2) ∧ 
    (∀ θ : ℝ, x θ = y θ ∨ x θ = -y θ) ∧
    (∀ ρ θ : ℝ, ρ * (Real.cos θ^2 - Real.sin θ^2) = 0 → 
      x θ = ρ * Real.cos θ ∧ y θ = ρ * Real.sin θ) :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_intersecting_lines_l2213_221369


namespace NUMINAMATH_CALUDE_complex_sum_power_l2213_221368

theorem complex_sum_power (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_power_l2213_221368


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_75_4500_l2213_221309

theorem gcd_lcm_sum_75_4500 : Nat.gcd 75 4500 + Nat.lcm 75 4500 = 4575 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_75_4500_l2213_221309


namespace NUMINAMATH_CALUDE_no_integer_solution_for_2007_l2213_221378

theorem no_integer_solution_for_2007 :
  ¬ ∃ (a b c : ℤ), a^2 + b^2 + c^2 = 2007 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_2007_l2213_221378


namespace NUMINAMATH_CALUDE_gcd_problem_l2213_221391

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) :
  Nat.gcd (Int.natAbs (4 * b^2 + 63 * b + 144)) (Int.natAbs (2 * b + 7)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2213_221391


namespace NUMINAMATH_CALUDE_count_integer_pairs_verify_count_l2213_221321

/-- The number of positive integer pairs (b,s) satisfying log₄(b²⁰s¹⁹⁰) = 4012 -/
theorem count_integer_pairs : Nat := by sorry

/-- Verifies that the count is correct -/
theorem verify_count : count_integer_pairs = 210 := by sorry

end NUMINAMATH_CALUDE_count_integer_pairs_verify_count_l2213_221321


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2213_221364

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_complement_equality :
  N ∩ (U \ M) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2213_221364


namespace NUMINAMATH_CALUDE_sum_range_for_distinct_positive_numbers_l2213_221346

theorem sum_range_for_distinct_positive_numbers (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_distinct : a ≠ b) 
  (h_eq : a^2 + a*b + b^2 = a + b) : 
  1 < a + b ∧ a + b < 4/3 := by
sorry

end NUMINAMATH_CALUDE_sum_range_for_distinct_positive_numbers_l2213_221346


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2213_221356

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℝ × ℝ := (-1, -2)

/-- First line equation: 2x + 3y + 8 = 0 -/
def line1 (p : ℝ × ℝ) : Prop :=
  2 * p.1 + 3 * p.2 + 8 = 0

/-- Second line equation: x - y - 1 = 0 -/
def line2 (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 - 1 = 0

/-- Theorem stating that the intersection_point satisfies both line equations
    and is the unique point that does so -/
theorem intersection_point_is_unique :
  line1 intersection_point ∧ 
  line2 intersection_point ∧ 
  ∀ p : ℝ × ℝ, line1 p ∧ line2 p → p = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2213_221356


namespace NUMINAMATH_CALUDE_trapezoid_KL_length_l2213_221363

/-- A trapezoid with points K and L on its diagonals -/
structure Trapezoid :=
  (A B C D K L : ℝ × ℝ)
  (is_trapezoid : sorry)
  (BC : ℝ)
  (AD : ℝ)
  (K_on_AC : sorry)
  (L_on_BD : sorry)
  (CK_KA_ratio : sorry)
  (BL_LD_ratio : sorry)

/-- The length of KL in the trapezoid -/
def KL_length (t : Trapezoid) : ℝ := sorry

theorem trapezoid_KL_length (t : Trapezoid) : 
  KL_length t = (1 / 11) * |7 * t.AD - 4 * t.BC| := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_KL_length_l2213_221363


namespace NUMINAMATH_CALUDE_inner_circle_radius_l2213_221338

theorem inner_circle_radius (s : ℝ) (h : s = 4) :
  let quarter_circle_radius := s / 2
  let square_diagonal := s * Real.sqrt 2
  let center_to_corner := square_diagonal / 2
  let r := (center_to_corner ^ 2 - quarter_circle_radius ^ 2).sqrt + quarter_circle_radius - center_to_corner
  r = 1 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l2213_221338


namespace NUMINAMATH_CALUDE_sum_mod_nine_l2213_221316

theorem sum_mod_nine : (2 + 44 + 666 + 8888 + 111110 + 13131312 + 1414141414) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l2213_221316


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l2213_221320

-- Define the types for lines and planes
variable (L : Type*) [AddCommGroup L] [Module ℝ L]
variable (P : Type*) [AddCommGroup P] [Module ℝ P]

-- Define the perpendicular and parallel relations
variable (perpendicular : L → P → Prop)
variable (parallel : L → L → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel
  (a b : L) (M : P)
  (h1 : perpendicular a M)
  (h2 : perpendicular b M) :
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l2213_221320


namespace NUMINAMATH_CALUDE_volume_of_second_cylinder_l2213_221399

/-- Given two cylinders with the same height and radii in the ratio 1:3, 
    if the volume of the first cylinder is 40 cc, 
    then the volume of the second cylinder is 360 cc. -/
theorem volume_of_second_cylinder 
  (h : ℝ) -- height of both cylinders
  (r₁ : ℝ) -- radius of the first cylinder
  (r₂ : ℝ) -- radius of the second cylinder
  (h_positive : h > 0)
  (r₁_positive : r₁ > 0)
  (ratio : r₂ = 3 * r₁) -- radii ratio condition
  (volume₁ : ℝ) -- volume of the first cylinder
  (h_volume₁ : volume₁ = Real.pi * r₁^2 * h) -- volume formula for the first cylinder
  (volume₁_value : volume₁ = 40) -- given volume of the first cylinder
  : Real.pi * r₂^2 * h = 360 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_second_cylinder_l2213_221399


namespace NUMINAMATH_CALUDE_tank_overflow_time_l2213_221300

/-- Represents the time it takes for a pipe to fill the tank -/
structure PipeRate where
  fill_time : ℝ
  fill_time_pos : fill_time > 0

/-- Represents the state of the tank filling process -/
structure TankFilling where
  overflow_time : ℝ
  pipe_a : PipeRate
  pipe_b : PipeRate
  pipe_b_close_time : ℝ

/-- The main theorem stating when the tank will overflow -/
theorem tank_overflow_time (tf : TankFilling) 
  (h1 : tf.pipe_a.fill_time = 2)
  (h2 : tf.pipe_b.fill_time = 1)
  (h3 : tf.pipe_b_close_time = tf.overflow_time - 0.5)
  (h4 : tf.overflow_time > 0) :
  tf.overflow_time = 1 := by
  sorry

#check tank_overflow_time

end NUMINAMATH_CALUDE_tank_overflow_time_l2213_221300


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2213_221329

/-- Theorem: For the quadratic equation x^2 + x - 2 = m, when m > 0, the equation has two distinct real roots. -/
theorem quadratic_equation_roots (m : ℝ) (h : m > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + x₁ - 2 = m ∧ x₂^2 + x₂ - 2 = m :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2213_221329


namespace NUMINAMATH_CALUDE_pythagorean_theorem_special_case_l2213_221313

/-- A right triangle with legs of lengths 1 and 2 -/
structure RightTriangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (is_right : leg1 = 1 ∧ leg2 = 2)

/-- The square of the hypotenuse of a right triangle -/
def hypotenuse_squared (t : RightTriangle) : ℝ :=
  t.leg1^2 + t.leg2^2

/-- Theorem: The square of the hypotenuse of a right triangle with legs 1 and 2 is 5 -/
theorem pythagorean_theorem_special_case (t : RightTriangle) :
  hypotenuse_squared t = 5 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_special_case_l2213_221313


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_half_l2213_221373

/-- Given a positive real number a, if the equation x² - 2ax - 2a ln x = 0
    has a unique solution in the interval (0, +∞), then a = 1/2. -/
theorem unique_solution_implies_a_half (a : ℝ) (ha : a > 0) :
  (∃! x : ℝ, x > 0 ∧ x^2 - 2*a*x - 2*a*(Real.log x) = 0) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_half_l2213_221373


namespace NUMINAMATH_CALUDE_probability_sum_12_l2213_221386

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The target sum we're aiming for -/
def targetSum : ℕ := 12

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (sum of 12) -/
def favorableOutcomes : ℕ := 10

/-- The probability of rolling a sum of 12 with three standard six-sided dice -/
theorem probability_sum_12 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 10 / 216 := by sorry

end NUMINAMATH_CALUDE_probability_sum_12_l2213_221386


namespace NUMINAMATH_CALUDE_archibald_apple_consumption_l2213_221395

def apples_per_day_first_two_weeks (x : ℝ) : Prop :=
  let first_two_weeks := 14 * x
  let next_three_weeks := 14 * x
  let last_two_weeks := 14 * 3
  let total_apples := 7 * 10
  first_two_weeks + next_three_weeks + last_two_weeks = total_apples

theorem archibald_apple_consumption : 
  ∃ x : ℝ, apples_per_day_first_two_weeks x ∧ x = 1 :=
sorry

end NUMINAMATH_CALUDE_archibald_apple_consumption_l2213_221395


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2213_221374

theorem polynomial_factorization (a b c : ℂ) :
  let ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)
  a^3 + b^3 + c^3 - 3*a*b*c = (a + b + c) * (a + ω*b + ω^2*c) * (a + ω^2*b + ω*c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2213_221374


namespace NUMINAMATH_CALUDE_solve_for_b_l2213_221303

-- Define the @ operation
def at_op (k : ℕ) (j : ℕ) : ℕ := (List.range j).foldl (λ acc i => acc * (k + i)) k

-- Define the problem parameters
def a : ℕ := 2020
def q : ℚ := 1/2

-- Theorem statement
theorem solve_for_b (b : ℕ) (h : (a : ℚ) / b = q) : b = 4040 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l2213_221303


namespace NUMINAMATH_CALUDE_final_result_proof_l2213_221359

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 740) : 
  (chosen_number / 4 : ℚ) - 175 = 10 := by
  sorry

end NUMINAMATH_CALUDE_final_result_proof_l2213_221359


namespace NUMINAMATH_CALUDE_nina_widget_purchase_l2213_221392

theorem nina_widget_purchase (initial_money : ℕ) (initial_widgets : ℕ) (price_reduction : ℕ) 
  (h1 : initial_money = 24)
  (h2 : initial_widgets = 6)
  (h3 : price_reduction = 1)
  : (initial_money / (initial_money / initial_widgets - price_reduction) : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_nina_widget_purchase_l2213_221392


namespace NUMINAMATH_CALUDE_product_mod_twenty_l2213_221353

theorem product_mod_twenty : (93 * 68 * 105) % 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_twenty_l2213_221353


namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l2213_221365

theorem no_solution_for_sock_problem : ¬ ∃ (m n : ℕ), 
  m + n = 2009 ∧ 
  (m^2 - m + n^2 - n : ℚ) / (2009 * 2008) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l2213_221365


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l2213_221344

noncomputable def f (x : ℝ) : ℝ := (Real.cos x + Real.sin x) * Real.cos (x - Real.pi / 2)

theorem monotone_increasing_interval (k : ℤ) :
  StrictMonoOn f { x | k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 } :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l2213_221344


namespace NUMINAMATH_CALUDE_simplify_2A_3B_value_2A_3B_specific_value_2A_3B_independent_l2213_221355

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3*x^2 - x + 2*y - 4*x*y
def B (x y : ℝ) : ℝ := 2*x^2 - 3*x - y + x*y

-- Theorem 1: Simplification of 2A - 3B
theorem simplify_2A_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y :=
sorry

-- Theorem 2: Value of 2A - 3B under specific conditions
theorem value_2A_3B_specific (x y : ℝ) (h1 : x + y = 6/7) (h2 : x*y = -1) :
  2 * A x y - 3 * B x y = 17 :=
sorry

-- Theorem 3: Value of 2A - 3B when independent of y
theorem value_2A_3B_independent :
  ∃ (x : ℝ), ∀ (y : ℝ), 2 * A x y - 3 * B x y = 49/11 :=
sorry

end NUMINAMATH_CALUDE_simplify_2A_3B_value_2A_3B_specific_value_2A_3B_independent_l2213_221355


namespace NUMINAMATH_CALUDE_terminating_decimal_expansion_of_13_375_l2213_221310

theorem terminating_decimal_expansion_of_13_375 :
  ∃ (n : ℕ) (k : ℕ), (13 : ℚ) / 375 = (34666 : ℚ) / 10^6 + k / (10^6 * 10^n) :=
sorry

end NUMINAMATH_CALUDE_terminating_decimal_expansion_of_13_375_l2213_221310


namespace NUMINAMATH_CALUDE_cos_shift_l2213_221390

theorem cos_shift (x : ℝ) : 
  2 * Real.cos (2 * (x - π / 8)) = 2 * Real.cos (2 * x - π / 4) := by
  sorry

#check cos_shift

end NUMINAMATH_CALUDE_cos_shift_l2213_221390


namespace NUMINAMATH_CALUDE_distance_between_points_l2213_221331

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 9)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l2213_221331


namespace NUMINAMATH_CALUDE_equation_solution_l2213_221393

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) (h3 : x ≠ 2) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (4 - x) / (x - 2) ↔ 
  x = -3 ∨ x = (1 + Real.sqrt 17) / 2 ∨ x = (1 - Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2213_221393


namespace NUMINAMATH_CALUDE_students_present_l2213_221381

/-- Given a class of 100 students with 14% absent, prove that the number of students present is 86. -/
theorem students_present (total_students : ℕ) (absent_percentage : ℚ) : 
  total_students = 100 → 
  absent_percentage = 14/100 → 
  (total_students : ℚ) * (1 - absent_percentage) = 86 :=
by sorry

end NUMINAMATH_CALUDE_students_present_l2213_221381


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_to_fraction_l2213_221328

theorem sqrt_mixed_number_to_fraction :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_to_fraction_l2213_221328


namespace NUMINAMATH_CALUDE_rectangular_to_polar_y_equals_x_l2213_221383

theorem rectangular_to_polar_y_equals_x :
  ∀ (x y ρ : ℝ) (θ : ℝ),
  (y = x) ↔ (θ = π / 4 ∧ ρ > 0) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_y_equals_x_l2213_221383


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_l2213_221317

def number : ℕ := 1386

theorem sum_of_largest_and_smallest_prime_factors :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ 
    largest.Prime ∧
    smallest ∣ number ∧ 
    largest ∣ number ∧
    (∀ p : ℕ, p.Prime → p ∣ number → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ number → p ≥ smallest) ∧
    smallest + largest = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_l2213_221317


namespace NUMINAMATH_CALUDE_incorrect_inequality_l2213_221332

theorem incorrect_inequality (a b : ℝ) (ha : -1 < a ∧ a < 2) (hb : -2 < b ∧ b < 3) :
  ¬ (∀ a b, -1 < a ∧ a < 2 ∧ -2 < b ∧ b < 3 → 2 < a * b ∧ a * b < 6) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l2213_221332


namespace NUMINAMATH_CALUDE_centers_form_rectangle_l2213_221385

/-- Represents a quadrilateral with side lengths -/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- Checks if a quadrilateral is inscribed -/
def is_inscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the center of a rectangle given two adjacent corners -/
def rectangle_center (p1 p2 : Point) (width height : ℝ) : Point :=
  sorry

/-- Checks if four points form a rectangle -/
def is_rectangle (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Main theorem: The centers of rectangles constructed on the sides of an inscribed quadrilateral form a rectangle -/
theorem centers_form_rectangle (q : Quadrilateral) (h : is_inscribed q) :
  let a := q.a
  let b := q.b
  let c := q.c
  let d := q.d
  let A := Point.mk 0 0  -- Arbitrary placement of A
  let B := Point.mk a 0  -- B is a units away from A on x-axis
  let C := sorry         -- C's position depends on the quadrilateral's shape
  let D := sorry         -- D's position depends on the quadrilateral's shape
  let P := rectangle_center A B a c
  let Q := rectangle_center B C b d
  let R := rectangle_center C D c a
  let S := rectangle_center D A d b
  is_rectangle P Q R S :=
sorry

end NUMINAMATH_CALUDE_centers_form_rectangle_l2213_221385


namespace NUMINAMATH_CALUDE_vector_operation_proof_l2213_221308

theorem vector_operation_proof (a b : ℝ × ℝ) :
  a = (2, 1) → b = (2, -2) → 2 • a - b = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l2213_221308


namespace NUMINAMATH_CALUDE_apples_to_friends_l2213_221358

theorem apples_to_friends (initial_apples : ℕ) (apples_left : ℕ) (apples_to_teachers : ℕ) (apples_eaten : ℕ) :
  initial_apples = 25 →
  apples_left = 3 →
  apples_to_teachers = 16 →
  apples_eaten = 1 →
  initial_apples - apples_left - apples_to_teachers - apples_eaten = 5 :=
by sorry

end NUMINAMATH_CALUDE_apples_to_friends_l2213_221358


namespace NUMINAMATH_CALUDE_root_equation_c_value_l2213_221343

theorem root_equation_c_value :
  ∀ (c d e : ℚ),
  (∃ (x : ℝ), x = -2 + 3 * Real.sqrt 5 ∧ x^4 + c*x^3 + d*x^2 + e*x - 48 = 0) →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_root_equation_c_value_l2213_221343


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2213_221354

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2 = 4) →
  (b / a = Real.sqrt 3) →
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2213_221354


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_triangle_area_l2213_221305

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/49 + y^2/24 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define perpendicularity condition
def is_perpendicular (m n : ℝ) : Prop := (n / (m + 5)) * (n / (m - 5)) = -1

-- Theorem statement
theorem ellipse_perpendicular_triangle_area (m n : ℝ) :
  is_on_ellipse m n → is_perpendicular m n →
  (1/2 : ℝ) * |10 * n| = 24 := by sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_triangle_area_l2213_221305


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2213_221318

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 5) * (x - 3) = -15 + k * x) ↔ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2213_221318


namespace NUMINAMATH_CALUDE_lynn_travel_time_l2213_221311

-- Define the problem parameters
def walk_fraction : ℚ := 1/3
def bike_fraction : ℚ := 2/3
def bike_speed_multiplier : ℚ := 4
def walk_time : ℚ := 9

-- Define the theorem
theorem lynn_travel_time :
  let bike_time := walk_time / bike_speed_multiplier
  walk_time + bike_time = 11.25 := by
  sorry


end NUMINAMATH_CALUDE_lynn_travel_time_l2213_221311


namespace NUMINAMATH_CALUDE_f_properties_l2213_221397

noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

theorem f_properties :
  ∃ (m : ℝ),
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≤ 6) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x m = 6) ∧
    (∀ x : ℝ, f x m ≥ 2) ∧
    (∀ k : ℤ, f (-Real.pi / 3 + k * Real.pi) m = 2) ∧
    m = 3 :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l2213_221397


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2213_221371

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x * y > 0) ∧
  (∃ x y : ℝ, x * y > 0 ∧ ¬(x > 0 ∧ y > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2213_221371


namespace NUMINAMATH_CALUDE_nancy_total_games_l2213_221327

/-- The number of games Nancy attended this month -/
def games_this_month : ℕ := 9

/-- The number of games Nancy attended last month -/
def games_last_month : ℕ := 8

/-- The number of games Nancy plans to attend next month -/
def games_next_month : ℕ := 7

/-- The total number of games Nancy would attend -/
def total_games : ℕ := games_this_month + games_last_month + games_next_month

theorem nancy_total_games : total_games = 24 := by
  sorry

end NUMINAMATH_CALUDE_nancy_total_games_l2213_221327


namespace NUMINAMATH_CALUDE_rectangle_area_l2213_221347

/-- Given a rectangle made from a wire of length 28 cm with a width of 6 cm, prove that its area is 48 cm². -/
theorem rectangle_area (wire_length : ℝ) (width : ℝ) (area : ℝ) :
  wire_length = 28 →
  width = 6 →
  area = (wire_length / 2 - width) * width →
  area = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2213_221347


namespace NUMINAMATH_CALUDE_range_of_g_l2213_221307

theorem range_of_g (x : ℝ) : -1 ≤ Real.sin x ^ 3 + Real.cos x ^ 2 ∧ Real.sin x ^ 3 + Real.cos x ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l2213_221307


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2213_221337

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 / (1 + Complex.I) → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2213_221337


namespace NUMINAMATH_CALUDE_mrs_thompson_potatoes_cost_l2213_221322

/-- Calculates the cost of potatoes given the number of chickens, cost per chicken, and total amount paid. -/
def cost_of_potatoes (num_chickens : ℕ) (cost_per_chicken : ℕ) (total_paid : ℕ) : ℕ :=
  total_paid - (num_chickens * cost_per_chicken)

/-- Proves that the cost of potatoes is 6 given the specific conditions of Mrs. Thompson's purchase. -/
theorem mrs_thompson_potatoes_cost :
  cost_of_potatoes 3 3 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_thompson_potatoes_cost_l2213_221322


namespace NUMINAMATH_CALUDE_total_spent_proof_l2213_221362

/-- The total amount spent on gifts and giftwrapping -/
def total_spent (gift_cost giftwrap_cost : ℚ) : ℚ :=
  gift_cost + giftwrap_cost

/-- Theorem: Given the cost of gifts and giftwrapping, prove the total amount spent -/
theorem total_spent_proof (gift_cost giftwrap_cost : ℚ) 
  (h1 : gift_cost = 561)
  (h2 : giftwrap_cost = 139) : 
  total_spent gift_cost giftwrap_cost = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_proof_l2213_221362


namespace NUMINAMATH_CALUDE_unique_monotonic_function_l2213_221345

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the property of being monotonic
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y ∨ f x > f y

-- Define the functional equation
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : PositiveReals, f (x * y) * f (f y / x) = 1

-- State the theorem
theorem unique_monotonic_function 
  (f : ℝ → ℝ) 
  (h1 : ∀ x : PositiveReals, f x > 0)
  (h2 : Monotonic f)
  (h3 : FunctionalEquation f) :
  ∀ x : PositiveReals, f x = 1 / x :=
sorry

end NUMINAMATH_CALUDE_unique_monotonic_function_l2213_221345
