import Mathlib

namespace NUMINAMATH_CALUDE_blood_expiration_time_l169_16926

-- Define the number of seconds in a day
def seconds_per_day : ℕ := 24 * 60 * 60

-- Define the expiration time in seconds (8!)
def expiration_time : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1

-- Define the donation time (noon)
def donation_hour : ℕ := 12

-- Theorem statement
theorem blood_expiration_time :
  (expiration_time / seconds_per_day = 0) ∧
  (expiration_time % seconds_per_day / 3600 + donation_hour = 23) :=
sorry

end NUMINAMATH_CALUDE_blood_expiration_time_l169_16926


namespace NUMINAMATH_CALUDE_simplify_fraction_l169_16945

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l169_16945


namespace NUMINAMATH_CALUDE_rainwater_farm_problem_l169_16917

/-- Mr. Rainwater's farm animals problem -/
theorem rainwater_farm_problem (goats cows chickens : ℕ) : 
  goats = 4 * cows →
  goats = 2 * chickens →
  chickens = 18 →
  cows = 9 := by
sorry

end NUMINAMATH_CALUDE_rainwater_farm_problem_l169_16917


namespace NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l169_16971

def dice_faces : ℕ := 6

def favorable_outcomes : ℕ := 6

def total_outcomes : ℕ := dice_faces * dice_faces

def probability_sum_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_seven_is_one_sixth : 
  probability_sum_seven = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l169_16971


namespace NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l169_16974

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l169_16974


namespace NUMINAMATH_CALUDE_problem_solution_l169_16908

theorem problem_solution : 
  (Real.sqrt 12 - 3 * Real.sqrt (1/3) + Real.sqrt 8 = Real.sqrt 3 + 2 * Real.sqrt 2) ∧ 
  ((Real.sqrt 5 - 1)^2 + Real.sqrt 5 * (Real.sqrt 5 + 2) = 11) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l169_16908


namespace NUMINAMATH_CALUDE_sqrt_x_plus_y_plus_five_halves_l169_16984

theorem sqrt_x_plus_y_plus_five_halves (x y : ℝ) : 
  y = Real.sqrt (2 * x - 3) + Real.sqrt (3 - 2 * x) + 5 →
  Real.sqrt (x + y + 5 / 2) = 3 ∨ Real.sqrt (x + y + 5 / 2) = -3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_y_plus_five_halves_l169_16984


namespace NUMINAMATH_CALUDE_nancy_future_games_l169_16907

/-- The number of games Nancy plans to attend next month -/
def games_next_month (games_this_month games_last_month total_games : ℕ) : ℕ :=
  total_games - (games_this_month + games_last_month)

/-- Proof that Nancy plans to attend 7 games next month -/
theorem nancy_future_games : games_next_month 9 8 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_future_games_l169_16907


namespace NUMINAMATH_CALUDE_triangle_area_proof_l169_16911

/-- Represents a triangle with parallel lines -/
structure TriangleWithParallelLines where
  /-- The area of the largest part -/
  largest_part_area : ℝ
  /-- The number of parallel lines -/
  num_parallel_lines : ℕ
  /-- The number of equal segments on the other two sides -/
  num_segments : ℕ
  /-- The number of parts the triangle is divided into -/
  num_parts : ℕ

/-- Theorem: If a triangle with 9 parallel lines dividing the sides into 10 equal segments
    has its largest part with an area of 38, then the total area of the triangle is 200 -/
theorem triangle_area_proof (t : TriangleWithParallelLines)
    (h1 : t.largest_part_area = 38)
    (h2 : t.num_parallel_lines = 9)
    (h3 : t.num_segments = 10)
    (h4 : t.num_parts = 10) :
    ∃ (total_area : ℝ), total_area = 200 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l169_16911


namespace NUMINAMATH_CALUDE_selection_theorem_l169_16921

/-- The number of students in the group -/
def total_students : ℕ := 7

/-- The number of representatives to be selected -/
def representatives : ℕ := 3

/-- The number of special students (A and B) -/
def special_students : ℕ := 2

/-- The number of ways to select 3 representatives from 7 students,
    with the condition that only one of students A and B is selected -/
def selection_ways : ℕ := Nat.choose special_students 1 * Nat.choose (total_students - special_students) (representatives - 1)

theorem selection_theorem : selection_ways = 20 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l169_16921


namespace NUMINAMATH_CALUDE_number_equation_l169_16990

theorem number_equation (x : ℝ) : 3 * x - 6 = 2 * x ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l169_16990


namespace NUMINAMATH_CALUDE_correct_bottles_calculation_l169_16924

/-- Given that B bottles of water can be purchased for P pennies,
    and 1 euro is worth 100 pennies, this function calculates
    the number of bottles that can be purchased for E euros. -/
def bottles_per_euro (B P E : ℚ) : ℚ :=
  (100 * E * B) / P

/-- Theorem stating that the number of bottles that can be purchased
    for E euros is (100 * E * B) / P, given the conditions. -/
theorem correct_bottles_calculation (B P E : ℚ) (hB : B > 0) (hP : P > 0) (hE : E > 0) :
  bottles_per_euro B P E = (100 * E * B) / P :=
by sorry

end NUMINAMATH_CALUDE_correct_bottles_calculation_l169_16924


namespace NUMINAMATH_CALUDE_bulletin_board_width_l169_16963

/-- Proves that a rectangular bulletin board with area 6400 cm² and length 160 cm has a width of 40 cm -/
theorem bulletin_board_width :
  ∀ (area length width : ℝ),
  area = 6400 ∧ length = 160 ∧ area = length * width →
  width = 40 := by
  sorry

end NUMINAMATH_CALUDE_bulletin_board_width_l169_16963


namespace NUMINAMATH_CALUDE_recess_time_calculation_l169_16964

/-- Calculates the total recess time based on grade distribution -/
def total_recess_time (base_time : ℕ) (a_count b_count c_count d_count : ℕ) : ℕ :=
  base_time + 2 * a_count + b_count - d_count

/-- Theorem stating that given the specific grade distribution, the total recess time is 47 minutes -/
theorem recess_time_calculation :
  let base_time : ℕ := 20
  let a_count : ℕ := 10
  let b_count : ℕ := 12
  let c_count : ℕ := 14
  let d_count : ℕ := 5
  total_recess_time base_time a_count b_count c_count d_count = 47 := by
  sorry

#eval total_recess_time 20 10 12 14 5

end NUMINAMATH_CALUDE_recess_time_calculation_l169_16964


namespace NUMINAMATH_CALUDE_max_grandchildren_l169_16929

/-- The number of grandchildren for a person with given children and grandchildren distribution -/
def grandchildren_count (num_children : ℕ) (num_children_with_same : ℕ) (num_grandchildren_same : ℕ) (num_children_different : ℕ) (num_grandchildren_different : ℕ) : ℕ :=
  (num_children_with_same * num_grandchildren_same) + (num_children_different * num_grandchildren_different)

/-- Theorem stating that Max has 58 grandchildren -/
theorem max_grandchildren :
  grandchildren_count 8 6 8 2 5 = 58 := by
  sorry

end NUMINAMATH_CALUDE_max_grandchildren_l169_16929


namespace NUMINAMATH_CALUDE_distinct_products_between_squares_l169_16965

theorem distinct_products_between_squares (n a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  n^2 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2 →
  a * d ≠ b * c :=
by sorry

end NUMINAMATH_CALUDE_distinct_products_between_squares_l169_16965


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l169_16986

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = x^2 - 2*x) :
  ∀ x < 0, f x = -x^2 - 2*x := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l169_16986


namespace NUMINAMATH_CALUDE_carpet_needed_proof_l169_16970

/-- Given a room with length and width, and an amount of existing carpet,
    calculate the additional carpet needed to cover the whole floor. -/
def additional_carpet_needed (length width existing_carpet : ℝ) : ℝ :=
  length * width - existing_carpet

/-- Proof that for a room of 4 feet by 20 feet with 18 square feet of existing carpet,
    62 square feet of additional carpet is needed. -/
theorem carpet_needed_proof :
  additional_carpet_needed 4 20 18 = 62 := by
  sorry

#eval additional_carpet_needed 4 20 18

end NUMINAMATH_CALUDE_carpet_needed_proof_l169_16970


namespace NUMINAMATH_CALUDE_g_502_solutions_l169_16925

-- Define g₁
def g₁ (x : ℚ) : ℚ := 1/2 - 4/(4*x + 2)

-- Define gₙ recursively
def g (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | 1 => g₁ x
  | n+1 => g₁ (g n x)

-- Theorem statement
theorem g_502_solutions (x : ℚ) : 
  g 502 x = x - 2 ↔ x = 115/64 ∨ x = 51/64 := by sorry

end NUMINAMATH_CALUDE_g_502_solutions_l169_16925


namespace NUMINAMATH_CALUDE_number_equation_proof_l169_16932

theorem number_equation_proof : ∃ n : ℝ, n + 11.95 - 596.95 = 3054 ∧ n = 3639 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l169_16932


namespace NUMINAMATH_CALUDE_building_E_floors_l169_16920

/-- The number of floors in Building A -/
def floors_A : ℕ := 4

/-- The number of floors in Building B -/
def floors_B : ℕ := floors_A + 9

/-- The number of floors in Building C -/
def floors_C : ℕ := 5 * floors_B - 6

/-- The number of floors in Building D -/
def floors_D : ℕ := 2 * floors_C - (floors_A + floors_B)

/-- The number of floors in Building E -/
def floors_E : ℕ := 3 * (floors_B + floors_C + floors_D) - 10

/-- Theorem stating that Building E has 509 floors -/
theorem building_E_floors : floors_E = 509 := by
  sorry

end NUMINAMATH_CALUDE_building_E_floors_l169_16920


namespace NUMINAMATH_CALUDE_alpha_value_theorem_l169_16916

/-- Given a function f(x) = x^α where α is a constant, 
    if the second derivative of f at x = -1 is 4, then α = -4 -/
theorem alpha_value_theorem (α : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^α) 
    (h2 : (deriv^[2] f) (-1) = 4) : 
  α = -4 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_theorem_l169_16916


namespace NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l169_16931

-- Define the number of songs
def num_songs : ℕ := 12

-- Define the length of the shortest song in seconds
def shortest_song : ℕ := 45

-- Define the common difference between song lengths
def song_length_diff : ℕ := 45

-- Define the length of the favorite song in seconds
def favorite_song_length : ℕ := 375

-- Define the total listening time in seconds
def total_listening_time : ℕ := 420

-- Function to calculate the length of the nth song
def song_length (n : ℕ) : ℕ := shortest_song + (n - 1) * song_length_diff

-- Theorem stating the probability of not hearing the entire favorite song
theorem probability_not_hearing_favorite_song :
  let total_orderings := num_songs.factorial
  let favorable_orderings := 3 * (num_songs - 1).factorial
  (total_orderings - favorable_orderings) / total_orderings = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l169_16931


namespace NUMINAMATH_CALUDE_least_possible_smallest_integer_l169_16993

theorem least_possible_smallest_integer
  (a b c d : ℤ) -- Four integers
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) -- They are distinct
  (h_average : (a + b + c + d) / 4 = 70) -- Their average is 70
  (h_largest : d = 90 ∧ d ≥ a ∧ d ≥ b ∧ d ≥ c) -- d is the largest and equals 90
  : a ≥ 184 -- The smallest integer is at least 184
:= by sorry

end NUMINAMATH_CALUDE_least_possible_smallest_integer_l169_16993


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l169_16937

def is_reducible (n : ℕ) : Prop :=
  (n > 15) ∧ (Nat.gcd (n - 15) (3 * n + 4) > 1)

theorem least_reducible_fraction :
  ∀ k : ℕ, k < 22 → ¬(is_reducible k) ∧ is_reducible 22 :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l169_16937


namespace NUMINAMATH_CALUDE_binomial_sum_one_l169_16991

theorem binomial_sum_one (a a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x, (a - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ = 80 →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_one_l169_16991


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l169_16936

theorem unique_solution_cube_equation :
  ∀ (x y z : ℤ), x^3 + 2*y^3 = 4*z^3 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l169_16936


namespace NUMINAMATH_CALUDE_solution_set_inequality_l169_16967

theorem solution_set_inequality (x : ℝ) : 
  (Set.Icc (-2 : ℝ) 3) = {x | (x - 1)^2 * (x + 2) * (x - 3) ≤ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l169_16967


namespace NUMINAMATH_CALUDE_longer_base_length_l169_16918

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithCircle where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Length of the shorter base -/
  short_base : ℝ
  /-- Length of the longer base -/
  long_base : ℝ
  /-- The circle is inscribed in the trapezoid -/
  inscribed : r > 0
  /-- The trapezoid is a right trapezoid -/
  right_angled : True
  /-- The shorter base is positive -/
  short_base_positive : short_base > 0
  /-- The longer base is longer than the shorter base -/
  base_inequality : long_base > short_base

/-- Theorem: The longer base of the trapezoid is 12 units -/
theorem longer_base_length (t : RightTrapezoidWithCircle) 
  (h1 : t.r = 3) 
  (h2 : t.short_base = 4) : 
  t.long_base = 12 := by
  sorry

end NUMINAMATH_CALUDE_longer_base_length_l169_16918


namespace NUMINAMATH_CALUDE_construction_materials_cost_l169_16903

/-- Calculates the total amount paid for construction materials --/
def total_amount_paid (cement_bags : ℕ) (cement_price : ℚ) (cement_discount : ℚ)
                      (sand_lorries : ℕ) (sand_tons_per_lorry : ℕ) (sand_price_per_ton : ℚ)
                      (sand_tax : ℚ) : ℚ :=
  let cement_cost := cement_bags * cement_price
  let cement_discount_amount := cement_cost * cement_discount
  let cement_total := cement_cost - cement_discount_amount
  let sand_tons := sand_lorries * sand_tons_per_lorry
  let sand_cost := sand_tons * sand_price_per_ton
  let sand_tax_amount := sand_cost * sand_tax
  let sand_total := sand_cost + sand_tax_amount
  cement_total + sand_total

/-- The total amount paid for construction materials is $13,310 --/
theorem construction_materials_cost :
  total_amount_paid 500 10 (5/100) 20 10 40 (7/100) = 13310 := by
  sorry

end NUMINAMATH_CALUDE_construction_materials_cost_l169_16903


namespace NUMINAMATH_CALUDE_stratified_sample_size_l169_16950

-- Define the total number of male and female athletes
def total_male : ℕ := 42
def total_female : ℕ := 30

-- Define the number of female athletes in the sample
def sampled_female : ℕ := 5

-- Theorem statement
theorem stratified_sample_size :
  ∃ (n : ℕ), 
    -- The sample size is the sum of sampled males and females
    n = (total_male * sampled_female / total_female) + sampled_female ∧
    -- The sample maintains the same ratio as the population
    n * total_female = (total_male + total_female) * sampled_female ∧
    -- The sample size is 12
    n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l169_16950


namespace NUMINAMATH_CALUDE_x_power_2048_minus_reciprocal_l169_16915

theorem x_power_2048_minus_reciprocal (x : ℝ) (h : x - 1/x = Real.sqrt 3) :
  x^2048 - 1/x^2048 = 277526 := by
  sorry

end NUMINAMATH_CALUDE_x_power_2048_minus_reciprocal_l169_16915


namespace NUMINAMATH_CALUDE_excursion_existence_l169_16981

theorem excursion_existence (S : Finset Nat) (E : Finset (Finset Nat)) 
  (h1 : S.card = 20) 
  (h2 : ∀ e ∈ E, e.card > 0) 
  (h3 : ∀ e ∈ E, e ⊆ S) :
  ∃ e ∈ E, ∀ s ∈ e, (E.filter (λ f => s ∈ f)).card ≥ E.card / 20 := by
sorry


end NUMINAMATH_CALUDE_excursion_existence_l169_16981


namespace NUMINAMATH_CALUDE_b_payment_correct_l169_16988

/-- The payment for a job completed by three workers A, B, and C. -/
def total_payment : ℚ := 529

/-- The fraction of work completed by A and C together. -/
def work_ac : ℚ := 19 / 23

/-- Calculate the payment for worker B given the total payment and the fraction of work done by A and C. -/
def payment_b (total : ℚ) (work_ac : ℚ) : ℚ :=
  total * (1 - work_ac)

theorem b_payment_correct : payment_b total_payment work_ac = 92 := by
  sorry

end NUMINAMATH_CALUDE_b_payment_correct_l169_16988


namespace NUMINAMATH_CALUDE_selling_price_increase_for_3360_profit_max_profit_at_10_yuan_increase_l169_16909

/-- Represents the profit function for T-shirt sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 3000

/-- Represents the constraint for a specific profit -/
def profit_constraint (x : ℝ) : Prop := profit_function x = 3360

/-- Theorem: The selling price increase that results in a profit of 3360 yuan is 2 yuan -/
theorem selling_price_increase_for_3360_profit :
  ∃ x : ℝ, profit_constraint x ∧ x = 2 := by sorry

/-- Theorem: The maximum profit occurs when the selling price is increased by 10 yuan, resulting in a profit of 4000 yuan -/
theorem max_profit_at_10_yuan_increase :
  ∃ x : ℝ, x = 10 ∧ profit_function x = 4000 ∧ 
  ∀ y : ℝ, profit_function y ≤ profit_function x := by sorry

end NUMINAMATH_CALUDE_selling_price_increase_for_3360_profit_max_profit_at_10_yuan_increase_l169_16909


namespace NUMINAMATH_CALUDE_count_not_divisible_1200_l169_16935

def count_not_divisible (n : ℕ) : ℕ :=
  (n - 1) - ((n - 1) / 6 + (n - 1) / 8 - (n - 1) / 24)

theorem count_not_divisible_1200 :
  count_not_divisible 1200 = 900 := by
  sorry

end NUMINAMATH_CALUDE_count_not_divisible_1200_l169_16935


namespace NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l169_16944

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l169_16944


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l169_16983

/-- Given a line L1 with equation x - y + 1 = 0 and a point P (2, -4),
    prove that the line L2 passing through P and parallel to L1
    has the equation x - y - 6 = 0 -/
theorem parallel_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => x - y + 1 = 0
  let P : ℝ × ℝ := (2, -4)
  let L2 : ℝ → ℝ → Prop := λ x y => x - y - 6 = 0
  (∀ x y, L2 x y ↔ (x - y = 6)) ∧
  L2 P.1 P.2 ∧
  (∀ x1 y1 x2 y2, L1 x1 y1 ∧ L2 x2 y2 → x1 - y1 = x2 - y2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l169_16983


namespace NUMINAMATH_CALUDE_benny_seashells_l169_16978

theorem benny_seashells (initial_seashells : Real) (percentage_given : Real) 
  (h1 : initial_seashells = 66.5)
  (h2 : percentage_given = 75) :
  initial_seashells - (percentage_given / 100) * initial_seashells = 16.625 := by
  sorry

end NUMINAMATH_CALUDE_benny_seashells_l169_16978


namespace NUMINAMATH_CALUDE_decreasing_g_implies_a_bound_f_nonpositive_implies_a_bound_l169_16912

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) : ℝ := Real.log x + a * x - 1 / x + b

def g (x : ℝ) : ℝ := f a b x + 2 / x

theorem decreasing_g_implies_a_bound :
  (∀ x > 0, ∀ y > 0, x < y → g a b y < g a b x) →
  a ≤ -1/4 := by sorry

theorem f_nonpositive_implies_a_bound :
  (∀ x > 0, f a b x ≤ 0) →
  a ≤ 1 - b := by sorry

end NUMINAMATH_CALUDE_decreasing_g_implies_a_bound_f_nonpositive_implies_a_bound_l169_16912


namespace NUMINAMATH_CALUDE_rental_cost_11_days_l169_16987

/-- Calculates the total cost of a car rental given the rental duration, daily rate, and weekly rate. -/
def rental_cost (days : ℕ) (daily_rate : ℕ) (weekly_rate : ℕ) : ℕ :=
  let weeks := days / 7
  let remaining_days := days % 7
  weeks * weekly_rate + remaining_days * daily_rate

/-- Theorem stating that the rental cost for 11 days is $310 given the specified rates. -/
theorem rental_cost_11_days :
  rental_cost 11 30 190 = 310 := by
  sorry

end NUMINAMATH_CALUDE_rental_cost_11_days_l169_16987


namespace NUMINAMATH_CALUDE_odd_as_difference_of_squares_l169_16969

theorem odd_as_difference_of_squares :
  ∀ n : ℤ, Odd n → ∃ a b : ℤ, n = a^2 - b^2 :=
by sorry

end NUMINAMATH_CALUDE_odd_as_difference_of_squares_l169_16969


namespace NUMINAMATH_CALUDE_original_game_points_l169_16985

/-- The number of points in the original game -/
def P : ℕ := 60

/-- X can give Y 20 points in a game of P points -/
def X_gives_Y (p : ℕ) : Prop := p - 20 > 0

/-- X can give Z 30 points in a game of P points -/
def X_gives_Z (p : ℕ) : Prop := p - 30 > 0

/-- In a game of 120 points, Y can give Z 30 points -/
def Y_gives_Z_120 : Prop := 120 - 30 > 0

/-- The ratio of scores when Y and Z play against X is equal to their ratio in a 120-point game -/
def score_ratio (p : ℕ) : Prop := (p - 20) * 90 = (p - 30) * 120

theorem original_game_points :
  X_gives_Y P ∧ X_gives_Z P ∧ Y_gives_Z_120 ∧ score_ratio P → P = 60 :=
by sorry

end NUMINAMATH_CALUDE_original_game_points_l169_16985


namespace NUMINAMATH_CALUDE_complex_number_properties_l169_16959

/-- The complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 4*m) (m^2 - m - 6)

/-- Predicate for a complex number being in the third quadrant -/
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

/-- Predicate for a complex number being on the imaginary axis -/
def on_imaginary_axis (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Predicate for a complex number being on the line x - y + 3 = 0 -/
def on_line (z : ℂ) : Prop := z.re - z.im + 3 = 0

theorem complex_number_properties (m : ℝ) :
  (in_third_quadrant (z m) ↔ 0 < m ∧ m < 3) ∧
  (on_imaginary_axis (z m) ↔ m = 0 ∨ m = 4) ∧
  (on_line (z m) ↔ m = 3) := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l169_16959


namespace NUMINAMATH_CALUDE_age_difference_l169_16904

theorem age_difference (A B : ℕ) : B = 37 → A + 10 = 2 * (B - 10) → A - B = 7 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l169_16904


namespace NUMINAMATH_CALUDE_symmetry_axes_count_cube_symmetry_axes_count_tetrahedron_symmetry_axes_count_l169_16901

/-- The number of axes of symmetry in a cube -/
def cube_symmetry_axes : ℕ := 13

/-- The number of axes of symmetry in a regular tetrahedron -/
def tetrahedron_symmetry_axes : ℕ := 7

/-- Theorem stating the number of axes of symmetry for a cube and a regular tetrahedron -/
theorem symmetry_axes_count :
  (cube_symmetry_axes = 13) ∧ (tetrahedron_symmetry_axes = 7) := by
  sorry

/-- Theorem for the number of axes of symmetry in a cube -/
theorem cube_symmetry_axes_count : cube_symmetry_axes = 13 := by
  sorry

/-- Theorem for the number of axes of symmetry in a regular tetrahedron -/
theorem tetrahedron_symmetry_axes_count : tetrahedron_symmetry_axes = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axes_count_cube_symmetry_axes_count_tetrahedron_symmetry_axes_count_l169_16901


namespace NUMINAMATH_CALUDE_caseys_corn_rows_l169_16949

/-- Represents the problem of calculating the number of corn plant rows Casey can water --/
theorem caseys_corn_rows :
  let pump_rate : ℚ := 3  -- gallons per minute
  let pump_time : ℕ := 25  -- minutes
  let plants_per_row : ℕ := 15
  let water_per_plant : ℚ := 1/2  -- gallons
  let num_pigs : ℕ := 10
  let water_per_pig : ℚ := 4  -- gallons
  let num_ducks : ℕ := 20
  let water_per_duck : ℚ := 1/4  -- gallons
  
  let total_water : ℚ := pump_rate * pump_time
  let water_for_animals : ℚ := num_pigs * water_per_pig + num_ducks * water_per_duck
  let water_for_plants : ℚ := total_water - water_for_animals
  let num_plants : ℚ := water_for_plants / water_per_plant
  let num_rows : ℚ := num_plants / plants_per_row
  
  num_rows = 4 := by sorry

end NUMINAMATH_CALUDE_caseys_corn_rows_l169_16949


namespace NUMINAMATH_CALUDE_injective_function_equation_l169_16906

theorem injective_function_equation (f : ℝ → ℝ) (h_inj : Function.Injective f) :
  (∀ x y : ℝ, x ≠ y → f ((x + y) / (x - y)) = (f x + f y) / (f x - f y)) →
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_injective_function_equation_l169_16906


namespace NUMINAMATH_CALUDE_radical_simplification_l169_16962

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (45 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 675 * q * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l169_16962


namespace NUMINAMATH_CALUDE_vector_relations_l169_16972

def a : ℝ × ℝ := (6, 2)
def b : ℝ → ℝ × ℝ := λ k => (-2, k)

theorem vector_relations (k : ℝ) :
  (∃ (c : ℝ), b k = c • a) → k = -2/3 ∧
  (a.1 * (b k).1 + a.2 * (b k).2 = 0) → k = 6 ∧
  (a.1 * (b k).1 + a.2 * (b k).2 < 0 ∧ ¬∃ (c : ℝ), b k = c • a) → k < 6 ∧ k ≠ -2/3 :=
by sorry

end NUMINAMATH_CALUDE_vector_relations_l169_16972


namespace NUMINAMATH_CALUDE_cookies_bought_l169_16989

theorem cookies_bought (total_groceries cake_packs : ℕ) 
  (h1 : total_groceries = 14)
  (h2 : cake_packs = 12)
  (h3 : ∃ cookie_packs : ℕ, cookie_packs + cake_packs = total_groceries) :
  ∃ cookie_packs : ℕ, cookie_packs = 2 ∧ cookie_packs + cake_packs = total_groceries :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_bought_l169_16989


namespace NUMINAMATH_CALUDE_martha_blocks_found_l169_16953

/-- The number of blocks Martha found -/
def blocks_found (initial final : ℕ) : ℕ := final - initial

/-- Martha's initial number of blocks -/
def martha_initial : ℕ := 4

/-- Martha's final number of blocks -/
def martha_final : ℕ := 84

theorem martha_blocks_found : blocks_found martha_initial martha_final = 80 := by
  sorry

end NUMINAMATH_CALUDE_martha_blocks_found_l169_16953


namespace NUMINAMATH_CALUDE_x_value_l169_16927

theorem x_value (x y : ℝ) (h : x / (x - 1) = (y^2 + 3*y - 5) / (y^2 + 3*y - 7)) :
  x = (y^2 + 3*y - 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l169_16927


namespace NUMINAMATH_CALUDE_average_equation_solution_l169_16910

theorem average_equation_solution (y : ℚ) : 
  (1 / 3 : ℚ) * ((y + 10) + (5 * y + 4) + (3 * y + 12)) = 6 * y - 8 → y = 50 / 9 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l169_16910


namespace NUMINAMATH_CALUDE_system_solution_l169_16960

theorem system_solution : ∃! (x y : ℚ), 3 * x + 4 * y = 12 ∧ 9 * x - 12 * y = -24 ∧ x = 2/3 ∧ y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l169_16960


namespace NUMINAMATH_CALUDE_message_difference_l169_16954

/-- The number of messages sent by Lucia and Alina over three days -/
def total_messages : ℕ := 680

/-- The number of messages sent by Lucia on the first day -/
def lucia_day1 : ℕ := 120

/-- The number of messages sent by Alina on the first day -/
def alina_day1 : ℕ := lucia_day1 - 20

/-- Calculates the total number of messages sent over three days -/
def calculate_total (a : ℕ) : ℕ :=
  a + lucia_day1 +  -- Day 1
  (2 * a) + (lucia_day1 / 3) +  -- Day 2
  a + lucia_day1  -- Day 3

/-- Theorem stating that the difference between Lucia's and Alina's messages on the first day is 20 -/
theorem message_difference :
  calculate_total alina_day1 = total_messages ∧
  alina_day1 < lucia_day1 ∧
  lucia_day1 - alina_day1 = 20 :=
by sorry

end NUMINAMATH_CALUDE_message_difference_l169_16954


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l169_16976

/-- Represents the capital contribution and time invested by a partner --/
structure Investment where
  capital : ℕ
  months : ℕ

/-- Calculates the capital-months for an investment --/
def capitalMonths (inv : Investment) : ℕ := inv.capital * inv.months

theorem profit_sharing_ratio 
  (a_investment : Investment) 
  (b_investment : Investment) 
  (h1 : a_investment.capital = 3500) 
  (h2 : a_investment.months = 12) 
  (h3 : b_investment.capital = 10500) 
  (h4 : b_investment.months = 6) :
  (capitalMonths a_investment) / (capitalMonths a_investment).gcd (capitalMonths b_investment) = 2 ∧ 
  (capitalMonths b_investment) / (capitalMonths a_investment).gcd (capitalMonths b_investment) = 3 :=
sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_l169_16976


namespace NUMINAMATH_CALUDE_system_solution_l169_16982

theorem system_solution :
  ∀ x y z : ℝ,
  x + y + z = 13 →
  x^2 + y^2 + z^2 = 61 →
  x*y + x*z = 2*y*z →
  ((x = 4 ∧ y = 3 ∧ z = 6) ∨ (x = 4 ∧ y = 6 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l169_16982


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l169_16938

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧ 
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l169_16938


namespace NUMINAMATH_CALUDE_car_speed_problem_l169_16914

theorem car_speed_problem (average_speed : ℝ) (first_hour_speed : ℝ) (total_time : ℝ) :
  average_speed = 65 →
  first_hour_speed = 100 →
  total_time = 2 →
  (average_speed * total_time - first_hour_speed) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l169_16914


namespace NUMINAMATH_CALUDE_gracie_is_56_inches_tall_l169_16902

/-- Gracie's height in inches -/
def gracies_height : ℕ := 56

/-- Theorem stating Gracie's height is 56 inches -/
theorem gracie_is_56_inches_tall : gracies_height = 56 := by
  sorry

end NUMINAMATH_CALUDE_gracie_is_56_inches_tall_l169_16902


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l169_16947

/-- Given a parallelogram with consecutive side lengths 10, 5y+3, 12, and 4x-1, prove that x + y = 91/20 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (4 * x - 1 = 10) →   -- First pair of opposite sides
  (5 * y + 3 = 12) →   -- Second pair of opposite sides
  x + y = 91/20 := by sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l169_16947


namespace NUMINAMATH_CALUDE_f_properties_l169_16900

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 3 * 2^(-x)
  else if x < 0 then 3 * 2^x - 2^(-x)
  else 0

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x < 0, f x = 3 * 2^x - 2^(-x)) ∧  -- f(x) for x < 0
  f 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_properties_l169_16900


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l169_16946

/-- The volume of a tetrahedron with vertices on coordinate axes -/
theorem tetrahedron_volume (d e f : ℝ) : 
  d > 0 → e > 0 → f > 0 →  -- Positive coordinates
  d^2 + e^2 = 49 →         -- DE = 7
  e^2 + f^2 = 64 →         -- EF = 8
  f^2 + d^2 = 81 →         -- FD = 9
  (1/6 : ℝ) * d * e * f = 4 * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_volume_l169_16946


namespace NUMINAMATH_CALUDE_math_problem_l169_16951

theorem math_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ (h : a * b - a - 2 * b = 0), a + 2 * b ≥ 8) ∧
  (a^2 / b + b^2 / a ≥ a + b) ∧
  (∀ (h : 1 / (a + 1) + 1 / (b + 2) = 1 / 3), a * b + a + b ≥ 14 + 6 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_math_problem_l169_16951


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l169_16952

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 27) : 
  r - p = 34 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l169_16952


namespace NUMINAMATH_CALUDE_maryville_population_increase_l169_16968

/-- The average annual population increase in Maryville between 2000 and 2005 -/
def average_annual_increase (pop_2000 pop_2005 : ℕ) : ℚ :=
  (pop_2005 - pop_2000 : ℚ) / 5

/-- Theorem stating that the average annual population increase in Maryville between 2000 and 2005 is 3400 -/
theorem maryville_population_increase :
  average_annual_increase 450000 467000 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_maryville_population_increase_l169_16968


namespace NUMINAMATH_CALUDE_log_equation_solution_l169_16940

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * Real.log x / Real.log 3 = Real.log (6 * x) / Real.log 3 → x = (6 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l169_16940


namespace NUMINAMATH_CALUDE_expression_value_l169_16955

theorem expression_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (x * y + 1)) : 
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (x * y + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l169_16955


namespace NUMINAMATH_CALUDE_thermostat_changes_l169_16922

theorem thermostat_changes (initial_temp : ℝ) : 
  initial_temp = 40 →
  let doubled := initial_temp * 2
  let after_dad := doubled - 30
  let after_mom := after_dad * 0.7
  let final_temp := after_mom + 24
  final_temp = 59 := by sorry

end NUMINAMATH_CALUDE_thermostat_changes_l169_16922


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l169_16975

def C : Finset Nat := {67, 71, 73, 76, 85}

theorem smallest_prime_factor_in_C : 
  ∃ (n : Nat), n ∈ C ∧ (∀ m ∈ C, ∀ p q : Nat, Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q) ∧ n = 76 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l169_16975


namespace NUMINAMATH_CALUDE_coin_distribution_l169_16995

theorem coin_distribution (x y z : ℕ) : 
  x + 2*y + 5*z = 71 →  -- total value is 71 kopecks
  x = y →  -- number of 1-kopeck coins equals number of 2-kopeck coins
  x + y + z = 31 →  -- total number of coins is 31
  (x = 12 ∧ y = 12 ∧ z = 7) :=
by sorry

end NUMINAMATH_CALUDE_coin_distribution_l169_16995


namespace NUMINAMATH_CALUDE_transistors_2010_l169_16966

/-- Moore's law: number of transistors doubles every two years -/
def moores_law (years : ℕ) : ℕ → ℕ := fun n => n * 2^(years / 2)

/-- Number of transistors in 1995 -/
def transistors_1995 : ℕ := 2000000

/-- Years between 1995 and 2010 -/
def years_passed : ℕ := 15

theorem transistors_2010 :
  moores_law years_passed transistors_1995 = 256000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_2010_l169_16966


namespace NUMINAMATH_CALUDE_cubic_system_unique_solution_l169_16997

theorem cubic_system_unique_solution (x y : ℝ) 
  (h1 : x^3 = 2 - y) (h2 : y^3 = 2 - x) : x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_cubic_system_unique_solution_l169_16997


namespace NUMINAMATH_CALUDE_jars_to_fill_l169_16958

def stars_per_jar : ℕ := 85
def initial_stars : ℕ := 33
def additional_stars : ℕ := 307

theorem jars_to_fill :
  (initial_stars + additional_stars) / stars_per_jar = 4 :=
by sorry

end NUMINAMATH_CALUDE_jars_to_fill_l169_16958


namespace NUMINAMATH_CALUDE_sequence_problem_l169_16999

theorem sequence_problem (n : ℕ) (a_n : ℕ → ℕ) : 
  (∀ k, a_n k = 3 * k + 4) → a_n n = 13 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l169_16999


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l169_16957

def boat_problem (boat_speed : ℝ) (against_stream_distance : ℝ) : ℝ :=
  let stream_speed := boat_speed - against_stream_distance
  boat_speed + stream_speed

theorem boat_distance_along_stream 
  (boat_speed : ℝ) 
  (against_stream_distance : ℝ) 
  (h1 : boat_speed = 15) 
  (h2 : against_stream_distance = 9) : 
  boat_problem boat_speed against_stream_distance = 21 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_l169_16957


namespace NUMINAMATH_CALUDE_f_properties_l169_16994

-- Define the function f(x) = x³ - 3x² + 3
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem f_properties :
  -- 1. The tangent line at (1, f(1)) is 3x + y - 4 = 0
  (∀ x y : ℝ, y = f' 1 * (x - 1) + f 1 ↔ 3*x + y - 4 = 0) ∧
  -- 2. The function has exactly 3 zeros
  (∃! (a b c : ℝ), a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
  -- 3. The function is symmetric about the point (1, 1)
  (∀ x : ℝ, f (1 + x) - 1 = -(f (1 - x) - 1)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l169_16994


namespace NUMINAMATH_CALUDE_function_value_at_six_l169_16930

/-- Given a function f such that f(4x+2) = x^2 - x + 1 for all real x, prove that f(6) = 1/2 -/
theorem function_value_at_six (f : ℝ → ℝ) (h : ∀ x : ℝ, f (4 * x + 2) = x^2 - x + 1) : f 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_six_l169_16930


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l169_16928

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^14 - 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l169_16928


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l169_16992

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 5
  let c : ℝ := -4
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + 5*x₁ - 4 = 0 ∧ x₂^2 + 5*x₂ - 4 = 0 ∧ x₁ ≠ x₂ :=
by sorry

#check quadratic_equation_solution

end NUMINAMATH_CALUDE_quadratic_equation_solution_l169_16992


namespace NUMINAMATH_CALUDE_afternoon_fliers_fraction_l169_16977

theorem afternoon_fliers_fraction (total : ℕ) (morning_fraction : ℚ) (left_over : ℕ) 
  (h_total : total = 2000)
  (h_morning : morning_fraction = 1 / 10)
  (h_left : left_over = 1350) :
  (total - left_over - (morning_fraction * total)) / (total - (morning_fraction * total)) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_afternoon_fliers_fraction_l169_16977


namespace NUMINAMATH_CALUDE_circle_C_properties_l169_16956

/-- Circle C defined by the equation x^2 + y^2 - 2x + 4y - 4 = 0 --/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- The center of circle C --/
def center : ℝ × ℝ := (1, -2)

/-- The radius of circle C --/
def radius : ℝ := 3

/-- A line with slope 1 --/
def line_with_slope_1 (a b : ℝ) (x y : ℝ) : Prop := y - b = x - a

/-- Theorem stating the properties of circle C and the existence of special lines --/
theorem circle_C_properties :
  (∀ x y : ℝ, circle_C x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
  (∃ a b : ℝ, (line_with_slope_1 a b (0) (0) ∧ 
              (line_with_slope_1 a b (-4) (-4) ∨ line_with_slope_1 a b (1) (1)) ∧
              (∃ x₁ y₁ x₂ y₂ : ℝ, 
                circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
                line_with_slope_1 a b x₁ y₁ ∧ line_with_slope_1 a b x₂ y₂ ∧
                (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 * ((x₁ + x₂)/2)^2 + 4 * ((y₁ + y₂)/2)^2))) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l169_16956


namespace NUMINAMATH_CALUDE_system_solution_unique_l169_16913

theorem system_solution_unique :
  ∃! (x y : ℝ), (4 * x - 3 * y = 11) ∧ (2 * x + y = 13) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l169_16913


namespace NUMINAMATH_CALUDE_jack_book_sale_l169_16996

/-- Calculates the amount received from selling books after a year --/
def amount_received (books_per_month : ℕ) (cost_per_book : ℕ) (months : ℕ) (loss : ℕ) : ℕ :=
  books_per_month * months * cost_per_book - loss

/-- Proves that Jack received $500 from selling the books --/
theorem jack_book_sale : amount_received 3 20 12 220 = 500 := by
  sorry

end NUMINAMATH_CALUDE_jack_book_sale_l169_16996


namespace NUMINAMATH_CALUDE_symmetry_implies_coordinates_l169_16948

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

/-- Given points A(a,-2) and B(3,b) are symmetric with respect to the origin, prove a = -3 and b = 2 -/
theorem symmetry_implies_coordinates (a b : ℝ) 
  (h : symmetric_wrt_origin a (-2) 3 b) : a = -3 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_coordinates_l169_16948


namespace NUMINAMATH_CALUDE_actual_distance_travelled_l169_16923

/-- The actual distance travelled by a person, given two walking speeds and an additional distance condition. -/
theorem actual_distance_travelled (speed1 speed2 additional_distance : ℝ) 
  (h1 : speed1 = 10)
  (h2 : speed2 = 14)
  (h3 : additional_distance = 20)
  (h4 : (actual_distance / speed1) = ((actual_distance + additional_distance) / speed2)) :
  actual_distance = 50 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_travelled_l169_16923


namespace NUMINAMATH_CALUDE_bahs_equivalent_to_1000_yahs_l169_16961

/-- The number of bahs equivalent to one rah -/
def bah_per_rah : ℚ := 15 / 24

/-- The number of rahs equivalent to one yah -/
def rah_per_yah : ℚ := 9 / 15

/-- The number of bahs equivalent to 1000 yahs -/
def bahs_per_1000_yahs : ℚ := 1000 * rah_per_yah * bah_per_rah

theorem bahs_equivalent_to_1000_yahs : bahs_per_1000_yahs = 375 := by
  sorry

end NUMINAMATH_CALUDE_bahs_equivalent_to_1000_yahs_l169_16961


namespace NUMINAMATH_CALUDE_alpha_value_l169_16905

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).im = 0 ∧ (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).im = 0 ∧ (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) : 
  α = 6 - 3 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_alpha_value_l169_16905


namespace NUMINAMATH_CALUDE_pat_to_kate_ratio_l169_16919

-- Define the variables
def total_hours : ℕ := 117
def mark_extra_hours : ℕ := 65

-- Define the hours charged by each person as real numbers
variable (pat_hours kate_hours mark_hours : ℝ)

-- Define the conditions
axiom total_hours_sum : pat_hours + kate_hours + mark_hours = total_hours
axiom pat_to_mark_ratio : pat_hours = (1/3) * mark_hours
axiom mark_to_kate_diff : mark_hours = kate_hours + mark_extra_hours

-- Define the theorem
theorem pat_to_kate_ratio :
  (∃ r : ℝ, pat_hours = r * kate_hours) →
  pat_hours / kate_hours = 2 := by sorry

end NUMINAMATH_CALUDE_pat_to_kate_ratio_l169_16919


namespace NUMINAMATH_CALUDE_hearty_beads_count_l169_16980

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := 5

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The total number of beads Hearty has -/
def total_beads : ℕ := (blue_packages + red_packages) * beads_per_package

theorem hearty_beads_count : total_beads = 320 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l169_16980


namespace NUMINAMATH_CALUDE_cube_shadow_problem_l169_16973

/-- The shadow area function calculates the area of the shadow cast by a cube,
    excluding the area beneath the cube. -/
def shadow_area (cube_edge : ℝ) (light_height : ℝ) : ℝ := sorry

/-- The problem statement -/
theorem cube_shadow_problem (y : ℝ) : 
  shadow_area 2 y = 200 → 
  ⌊1000 * y⌋ = 6140 := by sorry

end NUMINAMATH_CALUDE_cube_shadow_problem_l169_16973


namespace NUMINAMATH_CALUDE_solution_set_inequality_l169_16934

theorem solution_set_inequality (x : ℝ) : 
  (2 * x - 3) / (x + 2) ≤ 1 ↔ -2 < x ∧ x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l169_16934


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l169_16943

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = -(1/2 : ℂ) * (1 + Complex.I)) : 
  z.im = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l169_16943


namespace NUMINAMATH_CALUDE_three_digit_45_arithmetic_sequence_l169_16998

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b = (a + c) / 2

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_45_arithmetic_sequence :
  ∀ n : ℕ, is_three_digit n →
            n % 45 = 0 →
            is_arithmetic_sequence (n / 100) ((n / 10) % 10) (n % 10) →
            (n = 135 ∨ n = 630 ∨ n = 765) :=
sorry

end NUMINAMATH_CALUDE_three_digit_45_arithmetic_sequence_l169_16998


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l169_16941

theorem no_real_roots_quadratic (c : ℝ) :
  (∀ x : ℝ, x^2 + x - c ≠ 0) → c < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l169_16941


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l169_16979

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 47 →
  (n * original_mean - n * decrement) / n = 153 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l169_16979


namespace NUMINAMATH_CALUDE_alcohol_mixture_concentration_l169_16942

theorem alcohol_mixture_concentration
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid_poured : ℝ)
  (final_vessel_capacity : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percentage = 50)
  (h5 : total_liquid_poured = 8)
  (h6 : final_vessel_capacity = 10)
  : (vessel1_capacity * vessel1_alcohol_percentage / 100 +
     vessel2_capacity * vessel2_alcohol_percentage / 100) /
    final_vessel_capacity * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_concentration_l169_16942


namespace NUMINAMATH_CALUDE_equation_solution_l169_16933

theorem equation_solution : 
  ∃! (x : ℝ), x > 0 ∧ (1/2) * (3*x^2 - 1) = (x^2 - 50*x - 10) * (x^2 + 25*x + 5) ∧ x = 25 + 2 * Real.sqrt 159 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l169_16933


namespace NUMINAMATH_CALUDE_tree_growth_rate_l169_16939

/-- Proves that a tree with given initial and final heights over a specific time period has a certain growth rate per week. -/
theorem tree_growth_rate 
  (initial_height : ℝ) 
  (final_height : ℝ) 
  (months : ℕ) 
  (weeks_per_month : ℕ) 
  (h1 : initial_height = 10)
  (h2 : final_height = 42)
  (h3 : months = 4)
  (h4 : weeks_per_month = 4) :
  (final_height - initial_height) / (months * weeks_per_month : ℝ) = 2 := by
  sorry

#check tree_growth_rate

end NUMINAMATH_CALUDE_tree_growth_rate_l169_16939
