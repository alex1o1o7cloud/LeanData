import Mathlib

namespace NUMINAMATH_CALUDE_debt_average_payment_l2461_246146

/-- Proves that the average payment for a debt with specific conditions is $465 --/
theorem debt_average_payment 
  (total_installments : ℕ) 
  (first_payment_count : ℕ) 
  (first_payment_amount : ℚ) 
  (additional_amount : ℚ) : 
  total_installments = 52 →
  first_payment_count = 8 →
  first_payment_amount = 410 →
  additional_amount = 65 →
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + additional_amount
  let total_amount := 
    (first_payment_count * first_payment_amount) + 
    (remaining_payment_count * remaining_payment_amount)
  total_amount / total_installments = 465 := by
  sorry

end NUMINAMATH_CALUDE_debt_average_payment_l2461_246146


namespace NUMINAMATH_CALUDE_max_batteries_produced_l2461_246161

/-- Represents the production capacity of a robot type -/
structure RobotCapacity where
  time_per_battery : ℕ
  num_robots : ℕ

/-- Calculates the number of batteries a robot type can produce in a given time -/
def batteries_produced (capacity : RobotCapacity) (total_time : ℕ) : ℕ :=
  (total_time / capacity.time_per_battery) * capacity.num_robots

/-- Theorem: The maximum number of batteries produced is limited by the lowest production capacity -/
theorem max_batteries_produced 
  (robot_a : RobotCapacity) 
  (robot_b : RobotCapacity) 
  (robot_c : RobotCapacity) 
  (total_time : ℕ) :
  robot_a.time_per_battery = 6 →
  robot_b.time_per_battery = 9 →
  robot_c.time_per_battery = 3 →
  robot_a.num_robots = 8 →
  robot_b.num_robots = 6 →
  robot_c.num_robots = 6 →
  total_time = 300 →
  min (batteries_produced robot_a total_time) 
      (min (batteries_produced robot_b total_time) 
           (batteries_produced robot_c total_time)) = 198 :=
by sorry

end NUMINAMATH_CALUDE_max_batteries_produced_l2461_246161


namespace NUMINAMATH_CALUDE_class_size_is_25_l2461_246165

/-- Represents the number of students in a class with preferences for French fries and burgers. -/
structure ClassPreferences where
  frenchFries : ℕ  -- Number of students who like French fries
  burgers : ℕ      -- Number of students who like burgers
  both : ℕ         -- Number of students who like both
  neither : ℕ      -- Number of students who like neither

/-- Calculates the total number of students in the class. -/
def totalStudents (prefs : ClassPreferences) : ℕ :=
  prefs.frenchFries + prefs.burgers + prefs.neither - prefs.both

/-- Theorem stating that given the specific preferences, the total number of students is 25. -/
theorem class_size_is_25 (prefs : ClassPreferences)
  (h1 : prefs.frenchFries = 15)
  (h2 : prefs.burgers = 10)
  (h3 : prefs.both = 6)
  (h4 : prefs.neither = 6) :
  totalStudents prefs = 25 := by
  sorry

#eval totalStudents { frenchFries := 15, burgers := 10, both := 6, neither := 6 }

end NUMINAMATH_CALUDE_class_size_is_25_l2461_246165


namespace NUMINAMATH_CALUDE_sufficiency_of_P_for_Q_l2461_246150

theorem sufficiency_of_P_for_Q :
  ∀ x : ℝ, x ≥ 0 → 2 * x + 1 / (2 * x + 1) ≥ 1 ∧
  ¬(∀ x : ℝ, 2 * x + 1 / (2 * x + 1) ≥ 1 → x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficiency_of_P_for_Q_l2461_246150


namespace NUMINAMATH_CALUDE_sports_equipment_pricing_and_discount_l2461_246124

theorem sports_equipment_pricing_and_discount (soccer_price basketball_price : ℝ)
  (h1 : 2 * soccer_price + 3 * basketball_price = 410)
  (h2 : 5 * soccer_price + 2 * basketball_price = 530)
  (h3 : ∃ discount_rate : ℝ, 
    discount_rate * (5 * soccer_price + 5 * basketball_price) = 680 ∧ 
    0 < discount_rate ∧ 
    discount_rate < 1) :
  soccer_price = 70 ∧ basketball_price = 90 ∧ 
  ∃ discount_rate : ℝ, discount_rate * (5 * 70 + 5 * 90) = 680 ∧ discount_rate = 0.85 := by
sorry

end NUMINAMATH_CALUDE_sports_equipment_pricing_and_discount_l2461_246124


namespace NUMINAMATH_CALUDE_tulip_ratio_l2461_246152

/-- Given the number of red tulips for eyes and smile, and the total number of tulips,
    prove that the ratio of yellow tulips in the background to red tulips in the smile is 9:1 -/
theorem tulip_ratio (red_tulips_per_eye : ℕ) (red_tulips_smile : ℕ) (total_tulips : ℕ) :
  red_tulips_per_eye = 8 →
  red_tulips_smile = 18 →
  total_tulips = 196 →
  (total_tulips - (2 * red_tulips_per_eye + red_tulips_smile)) / red_tulips_smile = 9 := by
  sorry

end NUMINAMATH_CALUDE_tulip_ratio_l2461_246152


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2461_246143

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {9, a-5, 1-a}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {9} → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2461_246143


namespace NUMINAMATH_CALUDE_sum_of_cubes_representable_l2461_246106

theorem sum_of_cubes_representable (a b : ℤ) 
  (h1 : ∃ (x1 y1 : ℤ), a = x1^2 + 3*y1^2) 
  (h2 : ∃ (x2 y2 : ℤ), b = x2^2 + 3*y2^2) : 
  ∃ (x3 y3 : ℤ), a^3 + b^3 = x3^2 + 3*y3^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_representable_l2461_246106


namespace NUMINAMATH_CALUDE_sum_is_integer_four_or_negative_four_l2461_246137

theorem sum_is_integer_four_or_negative_four 
  (x y z t : ℝ) 
  (h : x / (y + z + t) = y / (z + t + x) ∧ 
       y / (z + t + x) = z / (t + x + y) ∧ 
       z / (t + x + y) = t / (x + y + z)) : 
  (x + y) / (z + t) + (y + z) / (t + x) + 
  (z + t) / (x + y) + (t + x) / (y + z) = 4 ∨
  (x + y) / (z + t) + (y + z) / (t + x) + 
  (z + t) / (x + y) + (t + x) / (y + z) = -4 :=
by sorry

end NUMINAMATH_CALUDE_sum_is_integer_four_or_negative_four_l2461_246137


namespace NUMINAMATH_CALUDE_power_product_equality_l2461_246112

theorem power_product_equality : (-4 : ℝ)^2010 * (-0.25 : ℝ)^2011 = -0.25 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l2461_246112


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2461_246111

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l2461_246111


namespace NUMINAMATH_CALUDE_c_investment_is_81000_l2461_246158

/-- Calculates the investment of partner C in a partnership business -/
def calculate_c_investment (a_investment b_investment : ℕ) (total_profit c_profit : ℕ) : ℕ :=
  let total_investment_ab := a_investment + b_investment
  let c_investment := (c_profit * (total_investment_ab + c_profit * total_investment_ab / (total_profit - c_profit))) / total_profit
  c_investment

/-- Theorem: Given the specific investments and profits, C's investment is 81000 -/
theorem c_investment_is_81000 :
  calculate_c_investment 27000 72000 80000 36000 = 81000 := by
  sorry

end NUMINAMATH_CALUDE_c_investment_is_81000_l2461_246158


namespace NUMINAMATH_CALUDE_loan_interest_percentage_l2461_246159

theorem loan_interest_percentage 
  (loan_amount : ℝ) 
  (monthly_payment : ℝ) 
  (num_months : ℕ) 
  (h1 : loan_amount = 150)
  (h2 : monthly_payment = 15)
  (h3 : num_months = 11) : 
  (monthly_payment * num_months - loan_amount) / loan_amount * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_percentage_l2461_246159


namespace NUMINAMATH_CALUDE_three_in_range_of_f_l2461_246178

/-- The function f(x) = x^2 + bx - 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 1

/-- Theorem: For all real b, there exists a real x such that f(x) = 3 -/
theorem three_in_range_of_f (b : ℝ) : ∃ x : ℝ, f b x = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_in_range_of_f_l2461_246178


namespace NUMINAMATH_CALUDE_unique_solution_l2461_246122

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_solution : ∃! x : ℕ, x > 0 ∧ digit_product x = x^2 - 10*x - 22 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2461_246122


namespace NUMINAMATH_CALUDE_square_division_perimeter_l2461_246123

theorem square_division_perimeter (s : ℝ) (h1 : s > 0) : 
  let square_perimeter := 4 * s
  let rectangle_length := s
  let rectangle_width := s / 2
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  square_perimeter = 200 → rectangle_perimeter = 150 := by
  sorry

end NUMINAMATH_CALUDE_square_division_perimeter_l2461_246123


namespace NUMINAMATH_CALUDE_alice_book_payment_percentage_l2461_246177

/-- The percentage of the suggested retail price that Alice paid for a book -/
theorem alice_book_payment_percentage 
  (suggested_retail_price : ℝ)
  (marked_price : ℝ)
  (alice_paid : ℝ)
  (h1 : marked_price = 0.6 * suggested_retail_price)
  (h2 : alice_paid = 0.4 * marked_price) :
  alice_paid / suggested_retail_price = 0.24 := by
sorry

end NUMINAMATH_CALUDE_alice_book_payment_percentage_l2461_246177


namespace NUMINAMATH_CALUDE_smallest_integer_in_ratio_l2461_246180

theorem smallest_integer_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 72 →
  b = 3 * a →
  c = 4 * a →
  a = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_ratio_l2461_246180


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l2461_246142

/-- Given a box of pens, calculate the probability of selecting two non-defective pens. -/
theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 10) 
  (h2 : defective_pens = 3) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 7 / 15 := by
sorry


end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l2461_246142


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2461_246191

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 → volume = (surface_area / 6) ^ (3/2) → volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2461_246191


namespace NUMINAMATH_CALUDE_volleyball_basketball_soccer_arrangement_l2461_246141

def num_stadiums : ℕ := 4
def num_competitions : ℕ := 3

def total_arrangements : ℕ := num_stadiums ^ num_competitions

def arrangements_all_same : ℕ := num_stadiums

theorem volleyball_basketball_soccer_arrangement :
  total_arrangements - arrangements_all_same = 60 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_basketball_soccer_arrangement_l2461_246141


namespace NUMINAMATH_CALUDE_cube_surface_area_with_holes_eq_222_l2461_246195

/-- Calculates the entire surface area of a cube with holes, including inside surfaces -/
def cubeSurfaceAreaWithHoles (cubeEdge : ℝ) (holeEdge : ℝ) : ℝ :=
  let originalSurface := 6 * cubeEdge^2
  let holeArea := 6 * holeEdge^2
  let newExposedArea := 6 * 4 * holeEdge^2
  originalSurface - holeArea + newExposedArea

/-- The entire surface area of the cube with holes is 222 square meters -/
theorem cube_surface_area_with_holes_eq_222 :
  cubeSurfaceAreaWithHoles 5 2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_with_holes_eq_222_l2461_246195


namespace NUMINAMATH_CALUDE_paths_from_A_to_D_l2461_246117

/-- Represents a point in the network -/
inductive Point : Type
| A : Point
| B : Point
| C : Point
| D : Point

/-- Represents the number of direct paths between two points -/
def direct_paths (p q : Point) : ℕ :=
  match p, q with
  | Point.A, Point.B => 2
  | Point.B, Point.C => 2
  | Point.C, Point.D => 2
  | Point.A, Point.C => 1
  | _, _ => 0

/-- The total number of paths from A to D -/
def total_paths : ℕ := 10

theorem paths_from_A_to_D :
  total_paths = 
    (direct_paths Point.A Point.B * direct_paths Point.B Point.C * direct_paths Point.C Point.D) +
    (direct_paths Point.A Point.C * direct_paths Point.C Point.D) :=
by sorry

end NUMINAMATH_CALUDE_paths_from_A_to_D_l2461_246117


namespace NUMINAMATH_CALUDE_f_10_sqrt_3_l2461_246157

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_10_sqrt_3 (f : ℝ → ℝ) 
    (hodd : OddFunction f)
    (hperiod : ∀ x, f (x + 2) = -f x)
    (hunit : ∀ x ∈ Set.Icc 0 1, f x = 2 * x) :
    f (10 * Real.sqrt 3) = -1.36 := by
  sorry

end NUMINAMATH_CALUDE_f_10_sqrt_3_l2461_246157


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l2461_246172

theorem halfway_between_fractions :
  (1 / 8 : ℚ) + ((1 / 3 : ℚ) - (1 / 8 : ℚ)) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l2461_246172


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2461_246129

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2461_246129


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l2461_246128

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^13 + i^18 + i^23 + i^28 + i^33 = i := by sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l2461_246128


namespace NUMINAMATH_CALUDE_park_trees_after_planting_l2461_246155

/-- The number of walnut trees in the park after planting -/
def total_trees (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem stating that the total number of walnut trees after planting is 211 -/
theorem park_trees_after_planting :
  total_trees 107 104 = 211 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_after_planting_l2461_246155


namespace NUMINAMATH_CALUDE_unicorn_count_correct_l2461_246168

/-- The number of unicorns in the Enchanted Forest --/
def num_unicorns : ℕ := 6

/-- The number of flowers that bloom with each unicorn step --/
def flowers_per_step : ℕ := 4

/-- The length of the journey in kilometers --/
def journey_length : ℕ := 9

/-- The length of each unicorn step in meters --/
def step_length : ℕ := 3

/-- The total number of flowers that bloom during the journey --/
def total_flowers : ℕ := 72000

/-- Theorem stating that the number of unicorns is correct given the conditions --/
theorem unicorn_count_correct : 
  num_unicorns * flowers_per_step * (journey_length * 1000 / step_length) = total_flowers :=
by sorry

end NUMINAMATH_CALUDE_unicorn_count_correct_l2461_246168


namespace NUMINAMATH_CALUDE_words_lost_proof_l2461_246160

/-- The number of letters in the language --/
def num_letters : ℕ := 67

/-- The number of words lost due to prohibiting one letter --/
def words_lost : ℕ := 135

/-- Theorem stating the number of words lost due to prohibiting one letter --/
theorem words_lost_proof :
  (1 : ℕ) + -- One-letter words lost
  (num_letters + num_letters) -- Two-letter words lost
  = words_lost := by sorry

end NUMINAMATH_CALUDE_words_lost_proof_l2461_246160


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2461_246173

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  B = π/4 ∧ 
  b = Real.sqrt 10 ∧
  Real.cos C = 2 * Real.sqrt 5 / 5 →
  Real.sin A = 3 * Real.sqrt 10 / 10 ∧
  a = 3 * Real.sqrt 2 ∧
  1/2 * a * b * Real.sin C = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2461_246173


namespace NUMINAMATH_CALUDE_complex_power_modulus_l2461_246188

theorem complex_power_modulus : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l2461_246188


namespace NUMINAMATH_CALUDE_initial_flour_amount_l2461_246136

theorem initial_flour_amount (initial : ℕ) : 
  initial + 2 = 10 → initial = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_flour_amount_l2461_246136


namespace NUMINAMATH_CALUDE_cube_root_of_y_fourth_root_of_y_to_six_l2461_246147

theorem cube_root_of_y_fourth_root_of_y_to_six (y : ℝ) :
  (y * (y^6)^(1/4))^(1/3) = 5 → y = 5^(6/5) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_y_fourth_root_of_y_to_six_l2461_246147


namespace NUMINAMATH_CALUDE_proposition_analysis_l2461_246139

theorem proposition_analysis (a b : ℝ) : 
  (∃ a b, a * b > 0 ∧ (a ≤ 0 ∨ b ≤ 0)) ∧ 
  (∃ a b, (a ≤ 0 ∨ b ≤ 0) ∧ a * b > 0) ∧
  (∀ a b, a * b ≤ 0 → a ≤ 0 ∨ b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_analysis_l2461_246139


namespace NUMINAMATH_CALUDE_inequality_proof_l2461_246134

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a * b + b * c + c * a = 3) :
  (a^2 / (1 + b * c)) + (b^2 / (1 + c * a)) + (c^2 / (1 + a * b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2461_246134


namespace NUMINAMATH_CALUDE_max_value_x2_plus_y2_l2461_246184

theorem max_value_x2_plus_y2 (x y : ℝ) :
  5 * x^2 - 10 * x + 4 * y^2 = 0 →
  x^2 + y^2 ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x2_plus_y2_l2461_246184


namespace NUMINAMATH_CALUDE_complex_difference_of_eighth_powers_l2461_246130

theorem complex_difference_of_eighth_powers : (2 + Complex.I) ^ 8 - (2 - Complex.I) ^ 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_of_eighth_powers_l2461_246130


namespace NUMINAMATH_CALUDE_same_color_probability_l2461_246107

/-- The probability of drawing two balls of the same color with replacement -/
theorem same_color_probability (green red blue : ℕ) (h_green : green = 8) (h_red : red = 6) (h_blue : blue = 4) :
  let total := green + red + blue
  (green / total) ^ 2 + (red / total) ^ 2 + (blue / total) ^ 2 = 29 / 81 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l2461_246107


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l2461_246169

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (3/5)^2 + (2/7)^2 / ((2/9)^2 + (1/6)^2) = 28*a/(45*b)) :
  Real.sqrt a / Real.sqrt b = 2 * Real.sqrt 105 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l2461_246169


namespace NUMINAMATH_CALUDE_coefficient_x3y5_times_two_l2461_246179

theorem coefficient_x3y5_times_two (x y : ℝ) : 2 * (Finset.range 9).sum (λ k => if k = 5 then Nat.choose 8 k else 0) = 112 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_times_two_l2461_246179


namespace NUMINAMATH_CALUDE_ratio_calculation_l2461_246118

theorem ratio_calculation (A B C : ℚ) (h : A = 2 * B ∧ C = 4 * B) :
  (3 * A + 2 * B) / (4 * C - A) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l2461_246118


namespace NUMINAMATH_CALUDE_walking_speed_equation_l2461_246132

theorem walking_speed_equation (x : ℝ) 
  (h1 : x > 0) -- Xiao Wang's speed is positive
  (h2 : x + 1 > 0) -- Xiao Zhang's speed is positive
  : 
  (15 / x - 15 / (x + 1) = 1 / 2) ↔ 
  (15 / x = 15 / (x + 1) + 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_equation_l2461_246132


namespace NUMINAMATH_CALUDE_magicians_trick_possible_l2461_246138

/-- A deck of cards numbered from 1 to 100 -/
def Deck : Type := Fin 100

/-- A selection of three cards from the deck -/
structure Selection :=
  (card1 card2 card3 : Deck)

/-- The additional card chosen by the second magician -/
def AdditionalCard : Type := Deck

/-- A function representing the strategy of the second magician -/
def SecondMagicianStrategy : Selection → AdditionalCard := sorry

/-- A function representing the strategy of the first magician -/
def FirstMagicianStrategy : AdditionalCard → Deck → Deck → Deck → Selection := sorry

/-- The main theorem stating that the magicians can perform the trick -/
theorem magicians_trick_possible :
  ∃ (second_strategy : Selection → AdditionalCard)
    (first_strategy : AdditionalCard → Deck → Deck → Deck → Selection),
  ∀ (selection : Selection),
  let additional_card := second_strategy selection
  let guessed_selection := first_strategy additional_card selection.card1 selection.card2 selection.card3
  guessed_selection = selection := by
  sorry

end NUMINAMATH_CALUDE_magicians_trick_possible_l2461_246138


namespace NUMINAMATH_CALUDE_ceiling_sqrt_224_l2461_246174

theorem ceiling_sqrt_224 : ⌈Real.sqrt 224⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_224_l2461_246174


namespace NUMINAMATH_CALUDE_one_zero_of_sin_log_l2461_246115

open Real

noncomputable def f (x : ℝ) : ℝ := sin (log x)

theorem one_zero_of_sin_log (h : ∀ x, 1 < x → x < exp π → f x = 0 → x = exp π) :
  ∃! x, 1 < x ∧ x < exp π ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_zero_of_sin_log_l2461_246115


namespace NUMINAMATH_CALUDE_gray_area_trees_count_l2461_246193

/-- Represents a rectangle with trees -/
structure TreeRectangle where
  total_trees : ℕ
  white_area_trees : ℕ

/-- Represents the setup of three overlapping rectangles -/
structure ThreeRectangles where
  rect1 : TreeRectangle
  rect2 : TreeRectangle
  rect3 : TreeRectangle

/-- The total number of trees in the gray (overlapping) areas -/
def gray_area_trees (setup : ThreeRectangles) : ℕ :=
  setup.rect1.total_trees - setup.rect1.white_area_trees +
  setup.rect2.total_trees - setup.rect2.white_area_trees

/-- Theorem stating the total number of trees in the gray areas -/
theorem gray_area_trees_count (setup : ThreeRectangles)
  (h1 : setup.rect1.total_trees = 100)
  (h2 : setup.rect2.total_trees = 100)
  (h3 : setup.rect3.total_trees = 100)
  (h4 : setup.rect1.white_area_trees = 82)
  (h5 : setup.rect2.white_area_trees = 82) :
  gray_area_trees setup = 26 := by
  sorry

end NUMINAMATH_CALUDE_gray_area_trees_count_l2461_246193


namespace NUMINAMATH_CALUDE_unique_p_q_sum_l2461_246185

theorem unique_p_q_sum (p q : ℤ) : 
  p > 1 → q > 1 → 
  ∃ (k₁ k₂ : ℤ), (2*p - 1 = k₁ * q) ∧ (2*q - 1 = k₂ * p) →
  p + q = 8 := by
sorry

end NUMINAMATH_CALUDE_unique_p_q_sum_l2461_246185


namespace NUMINAMATH_CALUDE_right_triangle_altitude_segment_ratio_l2461_246156

theorem right_triangle_altitude_segment_ratio :
  ∀ (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0),
  a^2 + b^2 = c^2 →  -- right triangle condition
  a = 3 * b →        -- leg ratio condition
  ∃ (d e : ℝ), d > 0 ∧ e > 0 ∧ d + e = c ∧ d / e = 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_segment_ratio_l2461_246156


namespace NUMINAMATH_CALUDE_series_result_l2461_246100

def series_term (n : ℕ) : ℕ := n.factorial * (n + 2) - (n + 1).factorial * (n + 3) + (n + 2).factorial

def series_sum : ℕ → ℕ
  | 0 => 1
  | n + 1 => series_sum n + series_term (n + 1)

theorem series_result : series_sum 2009 = 1 := by
  sorry

end NUMINAMATH_CALUDE_series_result_l2461_246100


namespace NUMINAMATH_CALUDE_closest_multiple_of_17_to_2502_l2461_246108

theorem closest_multiple_of_17_to_2502 :
  ∀ k : ℤ, k ≠ 147 → |2502 - 17 * 147| ≤ |2502 - 17 * k| :=
sorry

end NUMINAMATH_CALUDE_closest_multiple_of_17_to_2502_l2461_246108


namespace NUMINAMATH_CALUDE_two_digit_number_formation_l2461_246197

theorem two_digit_number_formation (k : ℕ) 
  (h1 : k > 0)
  (h2 : k ≤ 9)
  (h3 : ∀ (S T : ℕ), S = 11 * T * (k - 1) → S / T = 22) :
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_formation_l2461_246197


namespace NUMINAMATH_CALUDE_counterexample_exists_l2461_246163

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2461_246163


namespace NUMINAMATH_CALUDE_f_difference_equals_690_l2461_246176

/-- Given a function f(x) = x^5 + 3x^3 + 7x, prove that f(3) - f(-3) = 690 -/
theorem f_difference_equals_690 : 
  let f : ℝ → ℝ := λ x ↦ x^5 + 3*x^3 + 7*x
  f 3 - f (-3) = 690 := by sorry

end NUMINAMATH_CALUDE_f_difference_equals_690_l2461_246176


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l2461_246170

/-- The focal distance of a hyperbola with equation x²/20 - y²/5 = 1 is 10 -/
theorem hyperbola_focal_distance : 
  ∃ (c : ℝ), c > 0 ∧ c^2 = 25 ∧ 2*c = 10 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l2461_246170


namespace NUMINAMATH_CALUDE_chads_rope_length_l2461_246183

/-- Given that the ratio of Joey's rope length to Chad's rope length is 8:3,
    and Joey's rope is 56 cm long, prove that Chad's rope length is 21 cm. -/
theorem chads_rope_length (joey_length : ℝ) (chad_length : ℝ) 
    (h1 : joey_length = 56)
    (h2 : joey_length / chad_length = 8 / 3) : 
  chad_length = 21 := by
  sorry

end NUMINAMATH_CALUDE_chads_rope_length_l2461_246183


namespace NUMINAMATH_CALUDE_hash_problem_l2461_246199

-- Define the operation #
def hash (a b : ℕ) : ℕ := 4*a^2 + 4*b^2 + 8*a*b

-- Theorem statement
theorem hash_problem (a b : ℕ) :
  hash a b = 100 ∧ (a + b) + 6 = 11 → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_hash_problem_l2461_246199


namespace NUMINAMATH_CALUDE_camel_cost_is_5200_l2461_246148

-- Define the cost of each animal
def camel_cost : ℝ := 5200
def elephant_cost : ℝ := 13000
def ox_cost : ℝ := 8666.67
def horse_cost : ℝ := 2166.67

-- Define the relationships between animal costs
axiom camel_horse_ratio : 10 * camel_cost = 24 * horse_cost
axiom horse_ox_ratio : ∃ x : ℕ, x * horse_cost = 4 * ox_cost
axiom ox_elephant_ratio : 6 * ox_cost = 4 * elephant_cost
axiom elephant_total_cost : 10 * elephant_cost = 130000

-- Theorem to prove
theorem camel_cost_is_5200 : camel_cost = 5200 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_is_5200_l2461_246148


namespace NUMINAMATH_CALUDE_grade_distribution_l2461_246154

theorem grade_distribution (total_students : ℕ) 
  (prob_A prob_B prob_C prob_D : ℝ) 
  (h1 : prob_A = 0.6 * prob_B)
  (h2 : prob_C = 1.3 * prob_B)
  (h3 : prob_D = 0.8 * prob_B)
  (h4 : prob_A + prob_B + prob_C + prob_D = 1)
  (h5 : total_students = 50) :
  ∃ (num_B : ℕ), num_B = 14 ∧ 
    (↑num_B : ℝ) / total_students = prob_B := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l2461_246154


namespace NUMINAMATH_CALUDE_sum_mod_nine_l2461_246102

theorem sum_mod_nine : (88135 + 88136 + 88137 + 88138 + 88139 + 88140) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l2461_246102


namespace NUMINAMATH_CALUDE_f_composition_of_3_l2461_246196

def f (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_composition_of_3 : f (f (f (f (f 3)))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_3_l2461_246196


namespace NUMINAMATH_CALUDE_some_students_not_club_members_l2461_246103

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (ClubMember : U → Prop)
variable (StudiesLate : U → Prop)

-- State the theorem
theorem some_students_not_club_members
  (h1 : ∃ x, Student x ∧ ¬StudiesLate x)
  (h2 : ∀ x, ClubMember x → StudiesLate x) :
  ∃ x, Student x ∧ ¬ClubMember x :=
by
  sorry


end NUMINAMATH_CALUDE_some_students_not_club_members_l2461_246103


namespace NUMINAMATH_CALUDE_fraction_product_squared_main_theorem_l2461_246194

theorem fraction_product_squared (a b c d : ℚ) : 
  (a / b) ^ 2 * (c / d) ^ 2 = (a * c / (b * d)) ^ 2 :=
by sorry

theorem main_theorem : (6 / 7) ^ 2 * (1 / 2) ^ 2 = 9 / 49 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_squared_main_theorem_l2461_246194


namespace NUMINAMATH_CALUDE_jasmine_purchase_cost_l2461_246120

/-- The cost calculation for Jasmine's purchase of coffee beans and milk. -/
theorem jasmine_purchase_cost :
  let coffee_beans_pounds : ℕ := 4
  let milk_gallons : ℕ := 2
  let coffee_bean_price_per_pound : ℚ := 5/2
  let milk_price_per_gallon : ℚ := 7/2
  let total_cost : ℚ := coffee_beans_pounds * coffee_bean_price_per_pound + milk_gallons * milk_price_per_gallon
  total_cost = 17 := by sorry

end NUMINAMATH_CALUDE_jasmine_purchase_cost_l2461_246120


namespace NUMINAMATH_CALUDE_kimikos_age_l2461_246125

theorem kimikos_age (kayla_age kimiko_age min_driving_age wait_time : ℕ) : 
  kayla_age = kimiko_age / 2 →
  min_driving_age = 18 →
  kayla_age + wait_time = min_driving_age →
  wait_time = 5 →
  kimiko_age = 26 := by
sorry

end NUMINAMATH_CALUDE_kimikos_age_l2461_246125


namespace NUMINAMATH_CALUDE_wall_photo_dimensions_l2461_246104

/-- Given a rectangular paper with width 12 inches surrounded by a wall photo 2 inches wide,
    if the area of the wall photo is 96 square inches,
    then the length of the rectangular paper is 2 inches. -/
theorem wall_photo_dimensions (paper_length : ℝ) : 
  (paper_length + 4) * 16 = 96 → paper_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_wall_photo_dimensions_l2461_246104


namespace NUMINAMATH_CALUDE_mohamed_donated_three_bags_l2461_246127

/-- The number of bags Leila donated -/
def leila_bags : ℕ := 2

/-- The number of toys in each of Leila's bags -/
def leila_toys_per_bag : ℕ := 25

/-- The number of toys in each of Mohamed's bags -/
def mohamed_toys_per_bag : ℕ := 19

/-- The difference between Mohamed's and Leila's toy donations -/
def toy_difference : ℕ := 7

/-- Calculates the total number of toys Leila donated -/
def leila_total_toys : ℕ := leila_bags * leila_toys_per_bag

/-- Calculates the total number of toys Mohamed donated -/
def mohamed_total_toys : ℕ := leila_total_toys + toy_difference

/-- The number of bags Mohamed donated -/
def mohamed_bags : ℕ := mohamed_total_toys / mohamed_toys_per_bag

theorem mohamed_donated_three_bags : mohamed_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_mohamed_donated_three_bags_l2461_246127


namespace NUMINAMATH_CALUDE_equation_solution_l2461_246182

theorem equation_solution : 
  ∀ x : ℝ, (x - 5)^2 = (1/16)⁻¹ ↔ x = 1 ∨ x = 9 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2461_246182


namespace NUMINAMATH_CALUDE_curves_intersection_equality_l2461_246110

-- Define the four curves
def C₁ (x y : ℝ) : Prop := x^2 - y^2 = x / (x^2 + y^2)
def C₂ (x y : ℝ) : Prop := 2*x*y + y / (x^2 + y^2) = 3
def C₃ (x y : ℝ) : Prop := x^3 - 3*x*y^2 + 3*y = 1
def C₄ (x y : ℝ) : Prop := 3*y*x^2 - 3*x - y^3 = 0

-- State the theorem
theorem curves_intersection_equality :
  ∀ (x y : ℝ), (C₁ x y ∧ C₂ x y) ↔ (C₃ x y ∧ C₄ x y) := by sorry

end NUMINAMATH_CALUDE_curves_intersection_equality_l2461_246110


namespace NUMINAMATH_CALUDE_probability_five_green_marbles_l2461_246140

def num_green_marbles : ℕ := 6
def num_purple_marbles : ℕ := 4
def total_marbles : ℕ := num_green_marbles + num_purple_marbles
def num_draws : ℕ := 8
def num_green_draws : ℕ := 5

def probability_green : ℚ := num_green_marbles / total_marbles
def probability_purple : ℚ := num_purple_marbles / total_marbles

def combinations : ℕ := Nat.choose num_draws num_green_draws

theorem probability_five_green_marbles :
  (combinations : ℚ) * probability_green ^ num_green_draws * probability_purple ^ (num_draws - num_green_draws) =
  56 * (6/10)^5 * (4/10)^3 :=
sorry

end NUMINAMATH_CALUDE_probability_five_green_marbles_l2461_246140


namespace NUMINAMATH_CALUDE_arithmetic_progression_contains_10_start_l2461_246101

/-- An infinite increasing arithmetic progression of natural numbers contains a number starting with 10 -/
theorem arithmetic_progression_contains_10_start (a d : ℕ) (h : 0 < d) :
  ∃ k : ℕ, ∃ m : ℕ, (a + k * d) = 10 * 10^m + (a + k * d - 10 * 10^m) ∧ 
    10 * 10^m ≤ (a + k * d) ∧ (a + k * d) < 11 * 10^m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_contains_10_start_l2461_246101


namespace NUMINAMATH_CALUDE_simplify_expression_l2461_246131

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2461_246131


namespace NUMINAMATH_CALUDE_tangent_product_equality_l2461_246166

theorem tangent_product_equality : 
  Real.tan (55 * π / 180) * Real.tan (65 * π / 180) * Real.tan (75 * π / 180) = Real.tan (85 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equality_l2461_246166


namespace NUMINAMATH_CALUDE_expanded_product_terms_l2461_246171

theorem expanded_product_terms (a b c : ℕ) (ha : a = 6) (hb : b = 7) (hc : c = 5) :
  a * b * c = 210 := by
  sorry

end NUMINAMATH_CALUDE_expanded_product_terms_l2461_246171


namespace NUMINAMATH_CALUDE_rotation_result_l2461_246190

-- Define the shapes
inductive Shape
  | SmallCircle
  | Triangle
  | Square
  | Pentagon

-- Define the rotation directions
inductive RotationDirection
  | Clockwise
  | Counterclockwise

-- Define the configuration of shapes
structure Configuration :=
  (smallCircle : ℝ)  -- Angle of rotation for small circle
  (triangle : ℝ)     -- Angle of rotation for triangle
  (pentagon : ℝ)     -- Angle of rotation for pentagon
  (overall : ℝ)      -- Overall rotation of the configuration

-- Define the rotation function
def rotate (shape : Shape) (angle : ℝ) (direction : RotationDirection) : ℝ :=
  match direction with
  | RotationDirection.Clockwise => angle
  | RotationDirection.Counterclockwise => -angle

-- Define the initial configuration
def initialConfig : Configuration :=
  { smallCircle := 0, triangle := 0, pentagon := 0, overall := 0 }

-- Define the final configuration after rotations
def finalConfig (initial : Configuration) : Configuration :=
  { smallCircle := initial.smallCircle + rotate Shape.SmallCircle 45 RotationDirection.Counterclockwise,
    triangle := initial.triangle + rotate Shape.Triangle 180 RotationDirection.Clockwise,
    pentagon := initial.pentagon + rotate Shape.Pentagon 120 RotationDirection.Clockwise,
    overall := initial.overall + rotate Shape.Square 90 RotationDirection.Clockwise }

-- Theorem statement
theorem rotation_result :
  let final := finalConfig initialConfig
  final.smallCircle = -45 ∧
  final.triangle = 180 ∧
  final.pentagon = 120 ∧
  final.overall = 90 :=
by sorry

end NUMINAMATH_CALUDE_rotation_result_l2461_246190


namespace NUMINAMATH_CALUDE_annual_growth_rate_annual_growth_rate_proof_l2461_246105

/-- Given a monthly growth rate, calculate the annual growth rate -/
theorem annual_growth_rate (P : ℝ) : ℝ := 
  (1 + P)^11 - 1

/-- The annual growth rate is equal to (1+P)^11 - 1, where P is the monthly growth rate -/
theorem annual_growth_rate_proof (P : ℝ) : 
  annual_growth_rate P = (1 + P)^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_annual_growth_rate_proof_l2461_246105


namespace NUMINAMATH_CALUDE_work_completion_time_proportional_l2461_246116

/-- If a person can complete a piece of work in a given number of days, 
    then the time needed to complete a multiple of that work is proportional. -/
theorem work_completion_time_proportional 
  (original_days : ℕ) (work_multiple : ℕ) :
  original_days = 9 →
  work_multiple = 3 →
  original_days * work_multiple = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_proportional_l2461_246116


namespace NUMINAMATH_CALUDE_larry_remaining_cards_l2461_246167

/-- Given that Larry has 352 cards initially and Dennis takes 47 cards away,
    prove that Larry will have 305 cards remaining. -/
theorem larry_remaining_cards (initial_cards : ℕ) (cards_taken : ℕ) :
  initial_cards = 352 →
  cards_taken = 47 →
  initial_cards - cards_taken = 305 := by
  sorry

end NUMINAMATH_CALUDE_larry_remaining_cards_l2461_246167


namespace NUMINAMATH_CALUDE_star_three_two_l2461_246109

/-- The star operation defined as a^3 + 3a^2b + 3ab^2 + b^3 -/
def star (a b : ℝ) : ℝ := a^3 + 3*a^2*b + 3*a*b^2 + b^3

/-- Theorem stating that 3 star 2 equals 125 -/
theorem star_three_two : star 3 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_star_three_two_l2461_246109


namespace NUMINAMATH_CALUDE_product_and_gcd_conditions_l2461_246153

theorem product_and_gcd_conditions (a b : ℕ+) : 
  a * b = 864 ∧ Nat.gcd a b = 6 ↔ (a = 6 ∧ b = 144) ∨ (a = 144 ∧ b = 6) ∨ (a = 18 ∧ b = 48) ∨ (a = 48 ∧ b = 18) := by
  sorry

end NUMINAMATH_CALUDE_product_and_gcd_conditions_l2461_246153


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2461_246181

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a function type for edge coloring
def EdgeColoring := Fin 6 → Fin 6 → Color

-- Main theorem
theorem monochromatic_triangle_exists (coloring : EdgeColoring) : 
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  ((coloring a b = coloring b c ∧ coloring b c = coloring a c) ∨
   (coloring a b = Color.Red ∧ coloring b c = Color.Red ∧ coloring a c = Color.Red) ∨
   (coloring a b = Color.Blue ∧ coloring b c = Color.Blue ∧ coloring a c = Color.Blue)) :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2461_246181


namespace NUMINAMATH_CALUDE_gain_percent_when_selling_price_twice_cost_price_l2461_246145

/-- If the selling price of an item is twice its cost price, then the gain percent is 100% -/
theorem gain_percent_when_selling_price_twice_cost_price 
  (cost : ℝ) (selling : ℝ) (h : selling = 2 * cost) : 
  (selling - cost) / cost * 100 = 100 :=
sorry

end NUMINAMATH_CALUDE_gain_percent_when_selling_price_twice_cost_price_l2461_246145


namespace NUMINAMATH_CALUDE_quadratic_sum_l2461_246149

/-- 
Given a quadratic function f(x) = -3x^2 + 18x + 108, 
there exist constants a, b, and c such that 
f(x) = a(x+b)^2 + c for all x, 
and a + b + c = 129
-/
theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (-3 * x^2 + 18 * x + 108 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 129) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2461_246149


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_universal_negation_of_implication_l2461_246144

-- 1. Negation of existence
theorem negation_of_existence :
  (¬ ∃ x : ℤ, x^2 - 2*x - 3 = 0) ↔ (∀ x : ℤ, x^2 - 2*x - 3 ≠ 0) :=
by sorry

-- 2. Negation of universal quantification
theorem negation_of_universal :
  (¬ ∀ x : ℝ, x^2 + 3 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 3 < 2*x) :=
by sorry

-- 3. Negation of implication
theorem negation_of_implication :
  (¬ (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2)) ↔
  (∃ x y : ℝ, (x ≤ 1 ∨ y ≤ 1) ∧ x + y ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_universal_negation_of_implication_l2461_246144


namespace NUMINAMATH_CALUDE_student_ticket_price_is_318_l2461_246121

/-- Calculates the price of a student ticket given the total number of tickets sold,
    total revenue, adult ticket price, number of adult tickets sold, and number of student tickets sold. -/
def student_ticket_price (total_tickets : ℕ) (total_revenue : ℚ) (adult_price : ℚ) 
                         (adult_tickets : ℕ) (student_tickets : ℕ) : ℚ :=
  (total_revenue - (adult_price * adult_tickets)) / student_tickets

/-- Proves that the student ticket price is $3.18 given the specified conditions. -/
theorem student_ticket_price_is_318 :
  student_ticket_price 846 3846 6 410 436 = 318/100 := by
  sorry

end NUMINAMATH_CALUDE_student_ticket_price_is_318_l2461_246121


namespace NUMINAMATH_CALUDE_lisa_pizza_meat_distribution_l2461_246198

/-- The number of pieces of meat on each slice of Lisa's pizza --/
def pieces_per_slice : ℕ :=
  let pepperoni : ℕ := 30
  let ham : ℕ := 2 * pepperoni
  let sausage : ℕ := pepperoni + 12
  let total_meat : ℕ := pepperoni + ham + sausage
  let num_slices : ℕ := 6
  total_meat / num_slices

theorem lisa_pizza_meat_distribution :
  pieces_per_slice = 22 := by
  sorry

end NUMINAMATH_CALUDE_lisa_pizza_meat_distribution_l2461_246198


namespace NUMINAMATH_CALUDE_corveus_sleep_deficit_l2461_246187

/-- Calculates the total sleep deficit for Corveus in a week --/
def corveusWeeklySleepDeficit : ℤ :=
  let weekdaySleep : ℤ := 5 * 5  -- 4 hours night sleep + 1 hour nap, for 5 days
  let weekendSleep : ℤ := 5 * 2  -- 5 hours night sleep for 2 days
  let daylightSavingAdjustment : ℤ := 1  -- Extra hour due to daylight saving
  let midnightAwakenings : ℤ := 2  -- Loses 1 hour twice a week
  let actualSleep : ℤ := weekdaySleep + weekendSleep + daylightSavingAdjustment - midnightAwakenings
  let recommendedSleep : ℤ := 6 * 7  -- 6 hours per day for 7 days
  recommendedSleep - actualSleep

/-- Theorem stating that Corveus's weekly sleep deficit is 8 hours --/
theorem corveus_sleep_deficit : corveusWeeklySleepDeficit = 8 := by
  sorry

end NUMINAMATH_CALUDE_corveus_sleep_deficit_l2461_246187


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2461_246162

-- Define the condition "1 < x < 2"
def condition (x : ℝ) : Prop := 1 < x ∧ x < 2

-- Define the statement "x < 2"
def statement (x : ℝ) : Prop := x < 2

-- Theorem: "1 < x < 2" is a sufficient but not necessary condition for "x < 2"
theorem sufficient_not_necessary :
  (∀ x : ℝ, condition x → statement x) ∧
  (∃ x : ℝ, statement x ∧ ¬condition x) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2461_246162


namespace NUMINAMATH_CALUDE_special_polynomial_value_l2461_246151

/-- A polynomial function of degree n satisfying f(k) = k/(k+1) for k = 0, 1, ..., n -/
def SpecialPolynomial (n : ℕ) (f : ℝ → ℝ) : Prop :=
  (∃ p : Polynomial ℝ, Polynomial.degree p = n ∧ f = p.eval) ∧
  (∀ k : ℕ, k ≤ n → f k = k / (k + 1))

/-- The main theorem stating the value of f(n+1) for a SpecialPolynomial -/
theorem special_polynomial_value (n : ℕ) (f : ℝ → ℝ) 
  (h : SpecialPolynomial n f) : 
  f (n + 1) = (n + 1 + (-1)^(n + 1)) / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_value_l2461_246151


namespace NUMINAMATH_CALUDE_consecutive_odd_product_square_l2461_246126

theorem consecutive_odd_product_square : 
  ∃ (n : ℤ), (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_square_l2461_246126


namespace NUMINAMATH_CALUDE_function_passes_through_first_and_fourth_quadrants_l2461_246192

-- Define the conditions
def condition (a b c k : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (b + c - a) / a = k ∧
  (a + c - b) / b = k ∧
  (a + b - c) / c = k

-- Define the function
def f (k : ℝ) (x : ℝ) : ℝ := k * x - k

-- Define what it means for a function to pass through a quadrant
def passes_through_first_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y > 0 ∧ f x = y

def passes_through_fourth_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y < 0 ∧ f x = y

-- The theorem to be proved
theorem function_passes_through_first_and_fourth_quadrants
  (a b c k : ℝ) (h : condition a b c k) :
  passes_through_first_quadrant (f k) ∧
  passes_through_fourth_quadrant (f k) := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_first_and_fourth_quadrants_l2461_246192


namespace NUMINAMATH_CALUDE_integers_between_cubes_l2461_246119

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(10.1 : ℝ)^3⌋ - ⌈(9.8 : ℝ)^3⌉ + 1) ∧ n = 89 := by sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l2461_246119


namespace NUMINAMATH_CALUDE_closest_fraction_l2461_246175

def medals_won : ℚ := 20 / 120

def options : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (x : ℚ), x ∈ options ∧ 
  ∀ (y : ℚ), y ∈ options → |x - medals_won| ≤ |y - medals_won| :=
by sorry

end NUMINAMATH_CALUDE_closest_fraction_l2461_246175


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l2461_246113

/-- Represents an ellipse with given properties -/
structure Ellipse where
  e : ℝ  -- eccentricity
  ab_length : ℝ  -- length of AB

/-- Represents a line that intersects the ellipse -/
structure IntersectingLine where
  k : ℝ  -- slope of the line y = kx + 2

/-- Main theorem about the ellipse and intersecting line -/
theorem ellipse_and_line_properties
  (ell : Ellipse)
  (line : IntersectingLine)
  (h_e : ell.e = Real.sqrt 6 / 3)
  (h_ab : ell.ab_length = 2 * Real.sqrt 3 / 3) :
  (∃ (a b : ℝ), a^2 / 3 + b^2 = 1) ∧  -- Ellipse equation
  (∃ (x y : ℝ), x^2 / 3 + y^2 = 1 ∧ y = line.k * x + 2) ∧  -- Line intersects ellipse
  (∃ (c d : ℝ × ℝ),
    (c.1 - d.1)^2 + (c.2 - d.2)^2 = (-1 - c.1)^2 + (0 - c.2)^2 ∧  -- Circle condition
    (c.1 - d.1)^2 + (c.2 - d.2)^2 = (-1 - d.1)^2 + (0 - d.2)^2 ∧
    c.2 = line.k * c.1 + 2 ∧
    d.2 = line.k * d.1 + 2) →
  line.k = 7 / 6 := by
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l2461_246113


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l2461_246164

theorem imaginary_part_of_one_plus_i_squared (i : ℂ) (h : i^2 = -1) :
  (Complex.im ((1 : ℂ) + i)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l2461_246164


namespace NUMINAMATH_CALUDE_age_sum_proof_l2461_246135

theorem age_sum_proof (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_proof_l2461_246135


namespace NUMINAMATH_CALUDE_stripe_area_on_cylinder_l2461_246186

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylinder 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 40)
  (h2 : stripe_width = 2)
  (h3 : revolutions = 3) :
  stripe_width * revolutions * π * diameter = 240 * π := by
sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylinder_l2461_246186


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l2461_246133

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 2, 3, 4}
def Q : Set Nat := {3, 4, 5}

theorem intersection_P_complement_Q : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l2461_246133


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2461_246189

theorem algebraic_expression_value (x y : ℝ) 
  (hx : x = Real.sqrt 5 + 2) 
  (hy : y = Real.sqrt 5 - 2) : 
  x^2 - y + x*y = 12 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2461_246189


namespace NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l2461_246114

/-- Given a cone whose lateral surface, when unfolded, forms a semicircle with an area of 2π,
    the volume of the cone is (√3/3)π. -/
theorem cone_volume_from_lateral_surface (l r h : ℝ) : 
  l > 0 ∧ r > 0 ∧ h > 0 ∧
  (1/2) * Real.pi * l^2 = 2 * Real.pi ∧  -- Area of semicircle is 2π
  2 * Real.pi * r = Real.pi * l ∧        -- Circumference of base equals arc length of semicircle
  h^2 + r^2 = l^2 →                      -- Pythagorean theorem
  (1/3) * Real.pi * r^2 * h = (Real.sqrt 3 / 3) * Real.pi := by
sorry


end NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l2461_246114
