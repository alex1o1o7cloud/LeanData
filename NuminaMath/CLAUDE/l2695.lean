import Mathlib

namespace NUMINAMATH_CALUDE_prime_square_sum_l2695_269550

theorem prime_square_sum (p q n : ℕ) : 
  Prime p → Prime q → n^2 = p^2 + q^2 + p^2 * q^2 → 
  ((p = 2 ∧ q = 3 ∧ n = 7) ∨ (p = 3 ∧ q = 2 ∧ n = 7)) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_sum_l2695_269550


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2695_269556

/-- A line passes through a point if the point's coordinates satisfy the line equation -/
def PassesThrough (m : ℝ) (x y : ℝ) : Prop := m * x - y + 3 = 0

/-- The theorem states that for all real numbers m, 
    the line mx - y + 3 = 0 passes through the point (0, 3) -/
theorem fixed_point_theorem : ∀ m : ℝ, PassesThrough m 0 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2695_269556


namespace NUMINAMATH_CALUDE_largest_number_l2695_269557

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def number_A : Nat := to_base_10 [8, 5] 9
def number_B : Nat := to_base_10 [2, 0, 0] 6
def number_C : Nat := to_base_10 [6, 8] 8
def number_D : Nat := 70

theorem largest_number :
  number_A > number_B ∧ number_A > number_C ∧ number_A > number_D := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2695_269557


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l2695_269560

theorem polar_coordinates_of_point (x y : ℝ) (h : (x, y) = (-2, -2 * Real.sqrt 3)) :
  ∃ (ρ θ : ℝ), ρ = 4 ∧ θ = (4 * π) / 3 ∧
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l2695_269560


namespace NUMINAMATH_CALUDE_equal_area_rectangles_length_l2695_269539

/-- Given two rectangles of equal area, where one rectangle has dimensions 12 inches by 10 inches,
    and the other has a width of 5 inches, prove that the length of the second rectangle is 24 inches. -/
theorem equal_area_rectangles_length (area jordan_length jordan_width carol_width : ℝ)
    (h1 : area = jordan_length * jordan_width)
    (h2 : jordan_length = 12)
    (h3 : jordan_width = 10)
    (h4 : carol_width = 5)
    (h5 : area = carol_width * (area / carol_width)) :
    area / carol_width = 24 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_length_l2695_269539


namespace NUMINAMATH_CALUDE_cos2α_plus_sin2α_for_point_l2695_269538

theorem cos2α_plus_sin2α_for_point (α : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos α = -3 ∧ r * Real.sin α = 4) →
  Real.cos (2 * α) + Real.sin (2 * α) = -31/25 := by
  sorry

end NUMINAMATH_CALUDE_cos2α_plus_sin2α_for_point_l2695_269538


namespace NUMINAMATH_CALUDE_orange_juice_distribution_l2695_269509

theorem orange_juice_distribution (pitcher_capacity : ℝ) (h : pitcher_capacity > 0) :
  let juice_volume := (2/3) * pitcher_capacity
  let num_cups := 8
  let juice_per_cup := juice_volume / num_cups
  juice_per_cup / pitcher_capacity = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_distribution_l2695_269509


namespace NUMINAMATH_CALUDE_second_smallest_number_l2695_269590

def digits : List Nat := [1, 5, 6, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n / 10 = 5) ∧ (n % 10 ∈ digits) ∧ (n / 10 ∈ digits)

def count_smaller (n : Nat) : Nat :=
  (digits.filter (λ d => d < n % 10)).length

theorem second_smallest_number :
  ∃ n : Nat, is_valid_number n ∧ count_smaller n = 1 ∧ n = 56 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_number_l2695_269590


namespace NUMINAMATH_CALUDE_greatest_sum_of_digits_l2695_269579

/-- Represents a time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Calculates the sum of digits for a given time -/
def sumOfDigits (t : Time) : Nat :=
  (t.hours / 10) + (t.hours % 10) + (t.minutes / 10) + (t.minutes % 10)

/-- States that 19:59 has the greatest sum of digits among all possible times -/
theorem greatest_sum_of_digits :
  ∀ t : Time, sumOfDigits t ≤ sumOfDigits ⟨19, 59, by norm_num, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_of_digits_l2695_269579


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_63_degree_l2695_269534

def complement (α : ℝ) : ℝ := 90 - α

def supplement (β : ℝ) : ℝ := 180 - β

theorem supplement_of_complement_of_63_degree :
  supplement (complement 63) = 153 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_63_degree_l2695_269534


namespace NUMINAMATH_CALUDE_man_money_calculation_l2695_269599

/-- Calculates the total amount of money given the number of Rs. 50 and Rs. 500 notes -/
def totalAmount (fiftyNotes : ℕ) (fiveHundredNotes : ℕ) : ℕ :=
  50 * fiftyNotes + 500 * fiveHundredNotes

theorem man_money_calculation (totalNotes : ℕ) (fiftyNotes : ℕ) 
  (h1 : totalNotes = 126)
  (h2 : fiftyNotes = 117)
  (h3 : totalNotes = fiftyNotes + (totalNotes - fiftyNotes)) :
  totalAmount fiftyNotes (totalNotes - fiftyNotes) = 10350 := by
  sorry

#eval totalAmount 117 9

end NUMINAMATH_CALUDE_man_money_calculation_l2695_269599


namespace NUMINAMATH_CALUDE_expo_arrangement_plans_l2695_269592

/-- Represents the number of volunteers --/
def total_volunteers : ℕ := 5

/-- Represents the number of pavilions to be assigned --/
def pavilions_to_assign : ℕ := 3

/-- Represents the number of volunteers who cannot be assigned to a specific pavilion --/
def restricted_volunteers : ℕ := 2

/-- Represents the total number of arrangement plans --/
def total_arrangements : ℕ := 36

/-- Theorem stating the total number of arrangement plans --/
theorem expo_arrangement_plans :
  (total_volunteers = 5) →
  (pavilions_to_assign = 3) →
  (restricted_volunteers = 2) →
  (total_arrangements = 36) :=
by sorry

end NUMINAMATH_CALUDE_expo_arrangement_plans_l2695_269592


namespace NUMINAMATH_CALUDE_regular_star_points_l2695_269558

/-- Represents an n-pointed regular star with alternating angles --/
structure RegularStar where
  n : ℕ
  A : ℕ → ℝ
  B : ℕ → ℝ
  A_congruent : ∀ i j, A i = A j
  B_congruent : ∀ i j, B i = B j
  angle_difference : ∀ i, B i - A i = 20

/-- Theorem stating that the only possible number of points for the given conditions is 18 --/
theorem regular_star_points (star : RegularStar) : star.n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_points_l2695_269558


namespace NUMINAMATH_CALUDE_max_volume_at_10cm_l2695_269501

/-- The side length of the original square sheet of metal in centimeters -/
def a : ℝ := 60

/-- The volume of the box as a function of the cut-out square's side length -/
def volume (x : ℝ) : ℝ := (a - 2*x)^2 * x

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := 3600 - 480*x + 12*x^2

theorem max_volume_at_10cm :
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧
  volume_derivative x = 0 ∧
  (∀ y : ℝ, y > 0 → y < a/2 → volume y ≤ volume x) ∧
  x = 10 :=
sorry

end NUMINAMATH_CALUDE_max_volume_at_10cm_l2695_269501


namespace NUMINAMATH_CALUDE_elena_max_flour_l2695_269544

/-- Represents the recipe and available ingredients for Elena's bread --/
structure BreadRecipe where
  butter_ratio : ℚ  -- Ratio of butter to flour (in ounces per cup)
  sugar_ratio : ℚ   -- Ratio of sugar to flour (in ounces per cup)
  available_butter : ℚ  -- Available butter in ounces
  available_sugar : ℚ   -- Available sugar in ounces

/-- Calculates the maximum cups of flour that can be used given the recipe and available ingredients --/
def max_flour (recipe : BreadRecipe) : ℚ :=
  min 
    (recipe.available_butter / recipe.butter_ratio)
    (recipe.available_sugar / recipe.sugar_ratio)

/-- Elena's specific bread recipe and available ingredients --/
def elena_recipe : BreadRecipe :=
  { butter_ratio := 3/4
  , sugar_ratio := 2/5
  , available_butter := 24
  , available_sugar := 30 }

/-- Theorem stating that the maximum number of cups of flour Elena can use is 32 --/
theorem elena_max_flour : 
  max_flour elena_recipe = 32 := by sorry

end NUMINAMATH_CALUDE_elena_max_flour_l2695_269544


namespace NUMINAMATH_CALUDE_smallest_number_of_pens_l2695_269571

theorem smallest_number_of_pens (pen_package_size : Nat) (pencil_package_size : Nat)
  (h1 : pen_package_size = 12)
  (h2 : pencil_package_size = 15) :
  Nat.lcm pen_package_size pencil_package_size = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_pens_l2695_269571


namespace NUMINAMATH_CALUDE_peters_contribution_l2695_269506

/-- Given four friends pooling money for a purchase, prove Peter's contribution --/
theorem peters_contribution (john quincy andrew peter : ℝ) : 
  john > 0 ∧ 
  peter = 2 * john ∧ 
  quincy = peter + 20 ∧ 
  andrew = 1.15 * quincy ∧ 
  john + peter + quincy + andrew = 1211 →
  peter = 370.80 := by
  sorry

end NUMINAMATH_CALUDE_peters_contribution_l2695_269506


namespace NUMINAMATH_CALUDE_cistern_width_is_six_l2695_269569

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total wet surface area of the cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: Given the dimensions and wet surface area, the width of the cistern is 6 meters --/
theorem cistern_width_is_six (c : Cistern) 
    (h_length : c.length = 8)
    (h_depth : c.depth = 1.25)
    (h_area : wetSurfaceArea c = 83) : 
  c.width = 6 := by
  sorry

end NUMINAMATH_CALUDE_cistern_width_is_six_l2695_269569


namespace NUMINAMATH_CALUDE_perimeter_of_remaining_figure_l2695_269551

/-- The perimeter of a rectangle after cutting out squares --/
def perimeter_after_cuts (length width num_cuts cut_size : ℕ) : ℕ :=
  2 * (length + width) + num_cuts * (4 * cut_size - 2 * cut_size)

/-- Theorem stating the perimeter of the remaining figure after cuts --/
theorem perimeter_of_remaining_figure :
  perimeter_after_cuts 40 30 10 5 = 240 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_remaining_figure_l2695_269551


namespace NUMINAMATH_CALUDE_determinant_roots_count_l2695_269576

theorem determinant_roots_count (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  let f : ℝ → ℝ := λ x => x^4 * (x^2 + p^2 + q^2 + r^2)
  (∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, f x = 0 ∧ ∀ y, f y = 0 → y ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_determinant_roots_count_l2695_269576


namespace NUMINAMATH_CALUDE_mountain_climbing_equivalence_l2695_269521

/-- Given the elevations of two mountains and the number of times one is climbed,
    calculate how many times the other mountain needs to be climbed to cover the same distance. -/
theorem mountain_climbing_equivalence 
  (hugo_elevation : ℕ) 
  (elevation_difference : ℕ) 
  (hugo_climbs : ℕ) : 
  hugo_elevation = 10000 →
  elevation_difference = 2500 →
  hugo_climbs = 3 →
  (hugo_elevation * hugo_climbs) / (hugo_elevation - elevation_difference) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mountain_climbing_equivalence_l2695_269521


namespace NUMINAMATH_CALUDE_friendly_seq_uniqueness_l2695_269574

/-- A sequence of strictly increasing natural numbers -/
def IncreasingSeq := ℕ → ℕ

/-- Two sequences are friendly if every natural number is represented exactly once as their product -/
def Friendly (a b : IncreasingSeq) : Prop :=
  ∀ n : ℕ, ∃! (i j : ℕ), n = a i * b j

/-- The theorem stating that one friendly sequence uniquely determines the other -/
theorem friendly_seq_uniqueness (a b c : IncreasingSeq) :
  Friendly a b → Friendly a c → b = c := by sorry

end NUMINAMATH_CALUDE_friendly_seq_uniqueness_l2695_269574


namespace NUMINAMATH_CALUDE_initial_profit_percentage_l2695_269540

/-- Given an article with cost price P and initial selling price S, 
    where S > P, if selling the article at 2S results in a 180% profit, 
    then the initial profit percentage was 40%. -/
theorem initial_profit_percentage 
  (P S : ℝ) 
  (h1 : S > P) 
  (h2 : (2 * S - P) / P = 1.8) : 
  (S - P) / P = 0.4 := by
sorry

end NUMINAMATH_CALUDE_initial_profit_percentage_l2695_269540


namespace NUMINAMATH_CALUDE_double_root_condition_l2695_269533

/-- The equation has a double root when k is either 3 or 1/3 -/
theorem double_root_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) / (x^2 - 1) - 1 / (x - 1) = k / (x + 1) ∧ 
   ∀ y : ℝ, (k - 1) / (y^2 - 1) - 1 / (y - 1) = k / (y + 1) → y = x) ↔ 
  (k = 3 ∨ k = 1/3) :=
sorry

end NUMINAMATH_CALUDE_double_root_condition_l2695_269533


namespace NUMINAMATH_CALUDE_grade_distribution_l2695_269580

theorem grade_distribution (n : Nat) (k : Nat) : 
  n = 12 ∧ k = 3 → 
  (k^n : Nat) - k * ((k-1)^n : Nat) + (k * (k-2)^n : Nat) = 519156 := by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l2695_269580


namespace NUMINAMATH_CALUDE_archer_arrow_cost_l2695_269517

/-- Represents the archer's arrow usage and costs -/
structure ArcherData where
  shots_per_week : ℕ
  recovery_rate : ℚ
  personal_expense_rate : ℚ
  personal_expense : ℚ

/-- Calculates the cost per arrow given the archer's data -/
def cost_per_arrow (data : ArcherData) : ℚ :=
  let total_cost := data.personal_expense / data.personal_expense_rate
  let arrows_lost := data.shots_per_week * (1 - data.recovery_rate)
  total_cost / arrows_lost

/-- Theorem stating that the cost per arrow is $5.50 given the specific conditions -/
theorem archer_arrow_cost :
  let data : ArcherData := {
    shots_per_week := 800,
    recovery_rate := 1/5,
    personal_expense_rate := 3/10,
    personal_expense := 1056
  }
  cost_per_arrow data = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_archer_arrow_cost_l2695_269517


namespace NUMINAMATH_CALUDE_three_distinct_zeros_l2695_269519

-- Define the piecewise function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp (-x) - 1/2
  else x^3 - 3*m*x - 2

-- Theorem statement
theorem three_distinct_zeros (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f m x = 0 ∧ f m y = 0 ∧ f m z = 0) ↔ m > 1 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_zeros_l2695_269519


namespace NUMINAMATH_CALUDE_ellipse_properties_l2695_269508

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the semi-major axis, semi-minor axis, and semi-focal distance
def semi_major_axis : ℝ := 5
def semi_minor_axis : ℝ := 4
def semi_focal_distance : ℝ := 3

-- Theorem statement
theorem ellipse_properties :
  (∀ x y : ℝ, ellipse_equation x y) →
  semi_major_axis = 5 ∧ semi_minor_axis = 4 ∧ semi_focal_distance = 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2695_269508


namespace NUMINAMATH_CALUDE_sets_equal_implies_a_value_l2695_269597

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | -1 ≤ x ∧ x ≤ a}
def B (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = x + 1}
def C (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = x^2}

-- State the theorem
theorem sets_equal_implies_a_value (a : ℝ) (h1 : a > -1) (h2 : B a = C a) :
  a = 0 ∨ a = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sets_equal_implies_a_value_l2695_269597


namespace NUMINAMATH_CALUDE_additional_week_rate_is_12_l2695_269503

/-- The daily rate for additional weeks in a student youth hostel -/
def additional_week_rate (first_week_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) : ℚ :=
  (total_cost - first_week_rate * 7) / (total_days - 7)

/-- Theorem stating that the additional week rate is $12.00 per day -/
theorem additional_week_rate_is_12 :
  additional_week_rate 18 23 318 = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_week_rate_is_12_l2695_269503


namespace NUMINAMATH_CALUDE_symmetry_axis_sine_function_l2695_269549

theorem symmetry_axis_sine_function (φ : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sin (3 * x + φ)) →
  (|φ| < Real.pi / 2) →
  (∀ x : ℝ, Real.sin (3 * x + φ) = Real.sin (3 * (3 * Real.pi / 2 - x) + φ)) →
  φ = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_sine_function_l2695_269549


namespace NUMINAMATH_CALUDE_product_of_conjugates_equals_one_l2695_269575

theorem product_of_conjugates_equals_one :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_conjugates_equals_one_l2695_269575


namespace NUMINAMATH_CALUDE_sum_of_permutations_divisible_by_digit_sum_l2695_269535

/-- A type representing a digit from 1 to 9 -/
def Digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- A function to calculate the sum of all permutations of a five-digit number -/
def sumOfPermutations (a b c d e : Digit) : ℕ :=
  24 * 11111 * (a.val + b.val + c.val + d.val + e.val)

/-- The theorem statement -/
theorem sum_of_permutations_divisible_by_digit_sum 
  (a b c d e : Digit) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                c ≠ d ∧ c ≠ e ∧ 
                d ≠ e) : 
  (sumOfPermutations a b c d e) % (a.val + b.val + c.val + d.val + e.val) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_permutations_divisible_by_digit_sum_l2695_269535


namespace NUMINAMATH_CALUDE_total_chocolate_bars_l2695_269511

/-- Represents the number of chocolate bars in a large box -/
def chocolateBarsInLargeBox (smallBoxes : ℕ) (barsPerSmallBox : ℕ) : ℕ :=
  smallBoxes * barsPerSmallBox

/-- Proves that the total number of chocolate bars in the large box is 500 -/
theorem total_chocolate_bars :
  chocolateBarsInLargeBox 20 25 = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolate_bars_l2695_269511


namespace NUMINAMATH_CALUDE_smoothie_proportion_l2695_269500

/-- Given that 13 smoothies can be made from 3 bananas, prove that 65 smoothies can be made from 15 bananas. -/
theorem smoothie_proportion (make_smoothie : ℕ → ℕ) 
    (h : make_smoothie 3 = 13) : make_smoothie 15 = 65 := by
  sorry

#check smoothie_proportion

end NUMINAMATH_CALUDE_smoothie_proportion_l2695_269500


namespace NUMINAMATH_CALUDE_third_month_sale_l2695_269587

def sale_problem (m1 m2 m4 m5 m6 avg : ℕ) : Prop :=
  let total := avg * 6
  let known_sum := m1 + m2 + m4 + m5 + m6
  total - known_sum = 6200

theorem third_month_sale :
  sale_problem 5420 5660 6350 6500 6470 6100 :=
sorry

end NUMINAMATH_CALUDE_third_month_sale_l2695_269587


namespace NUMINAMATH_CALUDE_a_explicit_formula_l2695_269570

def a : ℕ → ℤ
  | 0 => -1
  | 1 => -3
  | 2 => -5
  | 3 => 5
  | (n + 4) => 8 * a (n + 3) - 22 * a (n + 2) + 24 * a (n + 1) - 9 * a n

theorem a_explicit_formula (n : ℕ) :
  a n = 2 + n - 3^(n + 1) + n * 3^n :=
by sorry

end NUMINAMATH_CALUDE_a_explicit_formula_l2695_269570


namespace NUMINAMATH_CALUDE_tan_alpha_tan_beta_l2695_269505

theorem tan_alpha_tan_beta (α β : ℝ) 
  (h1 : (Real.cos (α - β))^2 - (Real.cos (α + β))^2 = 1/2)
  (h2 : (1 + Real.cos (2 * α)) * (1 + Real.cos (2 * β)) = 1/3) :
  Real.tan α * Real.tan β = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_tan_beta_l2695_269505


namespace NUMINAMATH_CALUDE_expression_in_terms_of_k_l2695_269572

theorem expression_in_terms_of_k (x y k : ℝ) (h : x ≠ y) 
  (hk : (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k) :
  (x^8 + y^8) / (x^8 - y^8) - (x^8 - y^8) / (x^8 + y^8) = 
    (k - 2)^2 * (k + 2)^2 / (4 * k * (k^2 + 4)) :=
by sorry

end NUMINAMATH_CALUDE_expression_in_terms_of_k_l2695_269572


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_6_12_plus_5_13_l2695_269553

theorem smallest_prime_divisor_of_6_12_plus_5_13 :
  (Nat.minFac (6^12 + 5^13) = 5) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_6_12_plus_5_13_l2695_269553


namespace NUMINAMATH_CALUDE_one_adult_in_family_l2695_269502

/-- Represents the cost of tickets for a family visit to an aquarium -/
structure AquariumTickets where
  adultPrice : ℕ
  childPrice : ℕ
  numChildren : ℕ
  totalCost : ℕ

/-- Calculates the number of adults in the family based on ticket prices and total cost -/
def calculateAdults (tickets : AquariumTickets) : ℕ :=
  (tickets.totalCost - tickets.childPrice * tickets.numChildren) / tickets.adultPrice

/-- Theorem stating that for the given ticket prices and family composition, there is 1 adult -/
theorem one_adult_in_family (tickets : AquariumTickets) 
  (h1 : tickets.adultPrice = 35)
  (h2 : tickets.childPrice = 20)
  (h3 : tickets.numChildren = 6)
  (h4 : tickets.totalCost = 155) : 
  calculateAdults tickets = 1 := by
  sorry

#eval calculateAdults { adultPrice := 35, childPrice := 20, numChildren := 6, totalCost := 155 }

end NUMINAMATH_CALUDE_one_adult_in_family_l2695_269502


namespace NUMINAMATH_CALUDE_total_feed_amount_l2695_269593

/-- The price per pound of the cheaper feed -/
def cheap_price : ℚ := 18 / 100

/-- The price per pound of the expensive feed -/
def expensive_price : ℚ := 53 / 100

/-- The desired price per pound of the mixed feed -/
def mixed_price : ℚ := 36 / 100

/-- The amount of cheaper feed used (in pounds) -/
def cheap_amount : ℚ := 17

/-- The theorem stating that the total amount of feed mixed is 35 pounds -/
theorem total_feed_amount : 
  ∃ (expensive_amount : ℚ),
    cheap_amount + expensive_amount = 35 ∧
    (cheap_amount * cheap_price + expensive_amount * expensive_price) / (cheap_amount + expensive_amount) = mixed_price :=
by sorry

end NUMINAMATH_CALUDE_total_feed_amount_l2695_269593


namespace NUMINAMATH_CALUDE_vector_computation_l2695_269531

theorem vector_computation :
  (4 : ℝ) • ![(-3 : ℝ), 5] - (3 : ℝ) • ![(-2 : ℝ), 6] = ![-6, 2] := by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l2695_269531


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2695_269573

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 2025 * a + 2030 * b = 2035)
  (eq2 : 2027 * a + 2032 * b = 2037) : 
  a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2695_269573


namespace NUMINAMATH_CALUDE_average_unchanged_with_double_inclusion_l2695_269536

theorem average_unchanged_with_double_inclusion (n : ℕ) (scores : Fin n → ℝ) :
  let original_avg := (Finset.sum Finset.univ (λ i => scores i)) / n
  let new_sum := (Finset.sum Finset.univ (λ i => scores i)) + 2 * original_avg
  let new_avg := new_sum / (n + 2)
  new_avg = original_avg :=
by sorry

end NUMINAMATH_CALUDE_average_unchanged_with_double_inclusion_l2695_269536


namespace NUMINAMATH_CALUDE_fraction_division_multiplication_l2695_269543

theorem fraction_division_multiplication : (5 / 6 : ℚ) / (2 / 3) * (4 / 9) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_multiplication_l2695_269543


namespace NUMINAMATH_CALUDE_log_equation_solution_l2695_269577

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ x + 4 > 0 ∧ 2*x + 8 > 0 ∧
  Real.log x + Real.log (x + 4) = Real.log (2*x + 8) :=
by
  -- The unique solution is x = 2
  use 2
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2695_269577


namespace NUMINAMATH_CALUDE_jesse_carpet_amount_l2695_269520

/-- The amount of carpet Jesse already has -/
def carpet_already_has (room_length room_width additional_carpet_needed : ℝ) : ℝ :=
  room_length * room_width - additional_carpet_needed

/-- Theorem: Jesse already has 16 square feet of carpet -/
theorem jesse_carpet_amount :
  carpet_already_has 11 15 149 = 16 := by
  sorry

end NUMINAMATH_CALUDE_jesse_carpet_amount_l2695_269520


namespace NUMINAMATH_CALUDE_x_values_l2695_269528

theorem x_values (x : ℝ) :
  (x^3 - 3 = 3/8 → x = 3/2) ∧
  ((x - 1)^2 = 25 → x = 6 ∨ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_x_values_l2695_269528


namespace NUMINAMATH_CALUDE_remainder_problem_l2695_269523

theorem remainder_problem (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2695_269523


namespace NUMINAMATH_CALUDE_prism_volume_l2695_269518

/-- The volume of a right rectangular prism with face areas 18, 12, and 8 square inches -/
theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 18) 
  (h2 : y * z = 12) 
  (h3 : x * z = 8) : 
  x * y * z = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2695_269518


namespace NUMINAMATH_CALUDE_parallel_vector_sum_l2695_269598

/-- Given two vectors in ℝ², prove that if one is parallel to their sum, then the first component of the second vector is 1/2. -/
theorem parallel_vector_sum (a b : ℝ × ℝ) (h : a = (1, 2)) (h' : b.2 = 1) :
  (∃ (k : ℝ), b = k • (a + b)) → b.1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vector_sum_l2695_269598


namespace NUMINAMATH_CALUDE_simplify_expression_l2695_269568

theorem simplify_expression (m : ℝ) (h : m > 0) : 
  (Real.sqrt m * 3 * m) / ((6 * m) ^ 5) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2695_269568


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2695_269512

theorem smallest_candy_count : ∃ (n : ℕ), 
  n = 127 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬((m + 6) % 7 = 0 ∧ (m - 7) % 4 = 0)) ∧
  (n + 6) % 7 = 0 ∧ 
  (n - 7) % 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2695_269512


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2695_269589

/-- A function f: ℝ → ℝ is an "H function" if for any two distinct real numbers x₁ and x₂,
    the condition x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁ holds. -/
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function f: ℝ → ℝ is strictly increasing if for any two real numbers x₁ and x₂,
    x₁ < x₂ implies f x₁ < f x₂. -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

/-- Theorem: A function is an "H function" if and only if it is strictly increasing. -/
theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_H_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2695_269589


namespace NUMINAMATH_CALUDE_odd_sum_1_to_25_l2695_269565

theorem odd_sum_1_to_25 : 
  let odds := (List.range 13).map (fun i => 2 * i + 1)
  odds.sum = 169 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_1_to_25_l2695_269565


namespace NUMINAMATH_CALUDE_square_difference_l2695_269582

theorem square_difference (m : ℕ) : (m + 1)^2 - m^2 = 2*m + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2695_269582


namespace NUMINAMATH_CALUDE_no_intersection_implies_k_equals_three_l2695_269537

theorem no_intersection_implies_k_equals_three (k : ℕ+) :
  (∀ x y : ℝ, x^2 + y^2 = k^2 → x * y ≠ k) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_implies_k_equals_three_l2695_269537


namespace NUMINAMATH_CALUDE_sum_of_numbers_ge_04_l2695_269596

theorem sum_of_numbers_ge_04 : 
  let numbers : List ℚ := [4/5, 1/2, 9/10, 1/3]
  (numbers.filter (λ x => x ≥ 2/5)).sum = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_ge_04_l2695_269596


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2695_269561

theorem complex_equation_solution : 
  ∃ (z : ℂ), (5 : ℂ) + 2 * Complex.I * z = (1 : ℂ) - 6 * Complex.I * z ∧ z = Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2695_269561


namespace NUMINAMATH_CALUDE_B_largest_at_45_l2695_269541

/-- B_k is defined as the binomial coefficient (500 choose k) multiplied by 0.1^k -/
def B (k : ℕ) : ℝ := (Nat.choose 500 k : ℝ) * (0.1 ^ k)

/-- Theorem stating that B_k is largest when k = 45 -/
theorem B_largest_at_45 : ∀ k : ℕ, k ≤ 500 → B 45 ≥ B k := by
  sorry

end NUMINAMATH_CALUDE_B_largest_at_45_l2695_269541


namespace NUMINAMATH_CALUDE_correct_operation_l2695_269514

theorem correct_operation (x y : ℝ) : 4 * x^3 * y^2 - (-2)^2 * x^3 * y^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2695_269514


namespace NUMINAMATH_CALUDE_quadratic_equation_no_real_roots_l2695_269552

theorem quadratic_equation_no_real_roots 
  (p q a b c : ℝ) 
  (hp : p > 0) (hq : q > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hpq : p ≠ q)
  (hgeom : a^2 = p*q)  -- Geometric sequence condition
  (harith1 : b - p = c - b) (harith2 : c - b = q - c)  -- Arithmetic sequence conditions
  : (2*a)^2 - 4*b*c < 0 := by
  sorry

#check quadratic_equation_no_real_roots

end NUMINAMATH_CALUDE_quadratic_equation_no_real_roots_l2695_269552


namespace NUMINAMATH_CALUDE_power_multiplication_l2695_269532

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2695_269532


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2695_269547

/-- A function that checks if three numbers can form a right-angled triangle -/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

/-- The theorem stating that among the given sets, only {3, 4, 5} forms a right-angled triangle -/
theorem right_triangle_sets : 
  ¬ isRightTriangle 1 2 3 ∧
  isRightTriangle 3 4 5 ∧
  ¬ isRightTriangle 7 8 9 ∧
  ¬ isRightTriangle 5 10 20 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2695_269547


namespace NUMINAMATH_CALUDE_triangle_area_l2695_269504

/-- Proves that a triangle with the given conditions has an original area of 4 square cm -/
theorem triangle_area (base : ℝ) (h : ℝ → ℝ) :
  h 0 = 2 →
  h 1 = h 0 + 6 →
  (1 / 2 : ℝ) * base * h 1 - (1 / 2 : ℝ) * base * h 0 = 12 →
  (1 / 2 : ℝ) * base * h 0 = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l2695_269504


namespace NUMINAMATH_CALUDE_sin_neg_pi_sixth_l2695_269554

theorem sin_neg_pi_sixth : Real.sin (-π/6) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_pi_sixth_l2695_269554


namespace NUMINAMATH_CALUDE_cube_vertex_configurations_l2695_269594

/-- Represents a vertex of a cube -/
inductive CubeVertex
  | A | B | C | D | A1 | B1 | C1 | D1

/-- Represents a set of 4 vertices from a cube -/
def VertexSet := Finset CubeVertex

/-- Checks if a set of vertices forms a rectangle -/
def is_rectangle (vs : VertexSet) : Prop := sorry

/-- Checks if a set of vertices forms a tetrahedron with all equilateral triangle faces -/
def is_equilateral_tetrahedron (vs : VertexSet) : Prop := sorry

/-- Checks if a set of vertices forms a tetrahedron with all right triangle faces -/
def is_right_tetrahedron (vs : VertexSet) : Prop := sorry

/-- Checks if a set of vertices forms a tetrahedron with three isosceles right triangle faces and one equilateral triangle face -/
def is_mixed_tetrahedron (vs : VertexSet) : Prop := sorry

theorem cube_vertex_configurations :
  ∃ (vs1 vs2 vs3 vs4 : VertexSet),
    is_rectangle vs1 ∧
    is_equilateral_tetrahedron vs2 ∧
    is_right_tetrahedron vs3 ∧
    is_mixed_tetrahedron vs4 :=
  sorry

end NUMINAMATH_CALUDE_cube_vertex_configurations_l2695_269594


namespace NUMINAMATH_CALUDE_mean_of_data_is_10_l2695_269515

def data : List ℝ := [8, 12, 10, 11, 9]

theorem mean_of_data_is_10 :
  (data.sum / data.length : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_data_is_10_l2695_269515


namespace NUMINAMATH_CALUDE_december_spending_fraction_l2695_269595

def monthly_savings (month : Nat) : Rat :=
  match month with
  | 1 => 1/10
  | 2 => 3/25
  | 3 => 3/20
  | 4 => 1/5
  | m => if m ≤ 12 then (14 + m)/100 else 0

def total_savings : Rat :=
  (List.range 12).map (λ m => monthly_savings (m + 1)) |> List.sum

theorem december_spending_fraction :
  total_savings = 4 * (1 - monthly_savings 12) →
  1 - monthly_savings 12 = 39/50 := by
  sorry

end NUMINAMATH_CALUDE_december_spending_fraction_l2695_269595


namespace NUMINAMATH_CALUDE_triangle_angles_calculation_l2695_269586

-- Define the triangle and its properties
def Triangle (A B C : ℝ) (C_ext : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = 180 ∧
  C_ext = A + B

-- Theorem statement
theorem triangle_angles_calculation 
  (A B C C_ext : ℝ) 
  (h : Triangle A B C C_ext) 
  (hA : A = 64) 
  (hB : B = 33) 
  (hC_ext : C_ext = 120) :
  C = 83 ∧ ∃ D, D = 56 ∧ C_ext = A + D :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_calculation_l2695_269586


namespace NUMINAMATH_CALUDE_train_length_l2695_269591

/-- Calculates the length of a train given its speed and time to pass a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 10 → speed * time * (1000 / 3600) = 175 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2695_269591


namespace NUMINAMATH_CALUDE_expression_evaluation_l2695_269563

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator : ℚ := 5 + x * (5 + x) - 5^2
  let denominator : ℚ := x - 5 + x^2
  numerator / denominator = -26 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2695_269563


namespace NUMINAMATH_CALUDE_line_equation_l2695_269581

/-- Given a line l: ax + by + 1 = 0 and a circle x² + y² - 6y + 5 = 0, 
    this theorem proves that the line l is x - y + 3 = 0 
    if it's the axis of symmetry of the circle and perpendicular to x + y + 2 = 0 -/
theorem line_equation (a b : ℝ) : 
  (∀ x y : ℝ, a * x + b * y + 1 = 0 → 
    (x^2 + y^2 - 6*y + 5 = 0 → 
      (∃ c : ℝ, c * (a * x + b * y + 1) = x^2 + y^2 - 6*y + 5))) → 
  (a * 1 + b * 1 = -1) → 
  (a * x + b * y + 1 = 0 ↔ x - y + 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2695_269581


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l2695_269545

theorem bottle_cap_distribution (total_caps : ℕ) (total_boxes : ℕ) (caps_per_box : ℕ) : 
  total_caps = 60 → 
  total_boxes = 60 → 
  total_caps = total_boxes * caps_per_box → 
  caps_per_box = 1 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l2695_269545


namespace NUMINAMATH_CALUDE_third_plus_fifth_sum_l2695_269585

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  first_third_sum : a 1 + a 3 = 5
  common_ratio : q = 2
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem stating that a_3 + a_5 = 20 for the given geometric sequence -/
theorem third_plus_fifth_sum (seq : GeometricSequence) : seq.a 3 + seq.a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_plus_fifth_sum_l2695_269585


namespace NUMINAMATH_CALUDE_bicycle_route_length_l2695_269584

/-- The total length of a rectangular bicycle route -/
def total_length (horizontal_length vertical_length : ℝ) : ℝ :=
  2 * horizontal_length + 2 * vertical_length

/-- Theorem: The total length of the bicycle route is 52 km -/
theorem bicycle_route_length :
  let horizontal_length : ℝ := 13
  let vertical_length : ℝ := 13
  total_length horizontal_length vertical_length = 52 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_route_length_l2695_269584


namespace NUMINAMATH_CALUDE_garden_planting_area_l2695_269513

def garden_length : ℝ := 18
def garden_width : ℝ := 14
def pond_length : ℝ := 4
def pond_width : ℝ := 2
def flower_bed_base : ℝ := 3
def flower_bed_height : ℝ := 2

theorem garden_planting_area :
  garden_length * garden_width - (pond_length * pond_width + 1/2 * flower_bed_base * flower_bed_height) = 241 := by
  sorry

end NUMINAMATH_CALUDE_garden_planting_area_l2695_269513


namespace NUMINAMATH_CALUDE_expression_factorization_l2695_269522

theorem expression_factorization (x : ℝ) :
  (20 * x^3 + 100 * x - 10) - (-3 * x^3 + 5 * x - 15) = 5 * (23 * x^3 + 19 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2695_269522


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2695_269564

theorem max_value_of_expression (a b c d e f g h k : Int) 
  (ha : a = 1 ∨ a = -1) (hb : b = 1 ∨ b = -1) (hc : c = 1 ∨ c = -1) 
  (hd : d = 1 ∨ d = -1) (he : e = 1 ∨ e = -1) (hf : f = 1 ∨ f = -1) 
  (hg : g = 1 ∨ g = -1) (hh : h = 1 ∨ h = -1) (hk : k = 1 ∨ k = -1) : 
  (∀ a' b' c' d' e' f' g' h' k' : Int, 
    (a' = 1 ∨ a' = -1) → (b' = 1 ∨ b' = -1) → (c' = 1 ∨ c' = -1) → 
    (d' = 1 ∨ d' = -1) → (e' = 1 ∨ e' = -1) → (f' = 1 ∨ f' = -1) → 
    (g' = 1 ∨ g' = -1) → (h' = 1 ∨ h' = -1) → (k' = 1 ∨ k' = -1) → 
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' ≤ 4) ∧
  (∃ a' b' c' d' e' f' g' h' k' : Int, 
    (a' = 1 ∨ a' = -1) ∧ (b' = 1 ∨ b' = -1) ∧ (c' = 1 ∨ c' = -1) ∧ 
    (d' = 1 ∨ d' = -1) ∧ (e' = 1 ∨ e' = -1) ∧ (f' = 1 ∨ f' = -1) ∧ 
    (g' = 1 ∨ g' = -1) ∧ (h' = 1 ∨ h' = -1) ∧ (k' = 1 ∨ k' = -1) ∧
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' = 4) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2695_269564


namespace NUMINAMATH_CALUDE_parabola_equation_l2695_269588

/-- A parabola perpendicular to the x-axis passing through (1, -√2) has the equation y² = 2x -/
theorem parabola_equation : ∃ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ y^2 = 2*x) ∧ 
  (f 1 = -Real.sqrt 2) ∧
  (∀ x y : ℝ, f x = y → (x, y) ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1}) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2695_269588


namespace NUMINAMATH_CALUDE_parabola_points_ordering_l2695_269583

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 - 2*x + 2

/-- Point A on the parabola -/
def A : ℝ × ℝ := (-2, f (-2))

/-- Point B on the parabola -/
def B : ℝ × ℝ := (1, f 1)

/-- Point C on the parabola -/
def C : ℝ × ℝ := (2, f 2)

/-- y₁ is the y-coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ is the y-coordinate of point B -/
def y₂ : ℝ := B.2

/-- y₃ is the y-coordinate of point C -/
def y₃ : ℝ := C.2

theorem parabola_points_ordering : y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_ordering_l2695_269583


namespace NUMINAMATH_CALUDE_volume_of_smaller_cube_l2695_269525

/-- Given that eight equal-sized cubes form a larger cube with a surface area of 1536 cm²,
    prove that the volume of each smaller cube is 512 cm³. -/
theorem volume_of_smaller_cube (surface_area : ℝ) (num_small_cubes : ℕ) :
  surface_area = 1536 →
  num_small_cubes = 8 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    surface_area = 6 * side_length^2 ∧
    (side_length / 2)^3 = 512 :=
sorry

end NUMINAMATH_CALUDE_volume_of_smaller_cube_l2695_269525


namespace NUMINAMATH_CALUDE_building_block_length_l2695_269542

theorem building_block_length 
  (box_height box_width box_length : ℝ)
  (block_height block_width : ℝ)
  (num_blocks : ℕ) :
  box_height = 8 →
  box_width = 10 →
  box_length = 12 →
  block_height = 3 →
  block_width = 2 →
  num_blocks = 40 →
  ∃ (block_length : ℝ),
    box_height * box_width * box_length = 
    num_blocks * block_height * block_width * block_length ∧
    block_length = 4 :=
by sorry

end NUMINAMATH_CALUDE_building_block_length_l2695_269542


namespace NUMINAMATH_CALUDE_sequence_property_l2695_269559

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : ∀ (p q : ℕ), a (p + q) = a p + a q) 
  (h2 : a 2 = 4) : 
  a 9 = 18 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l2695_269559


namespace NUMINAMATH_CALUDE_rectangle_composition_l2695_269578

/-- The side length of the middle square in a specific rectangular arrangement -/
def square_side_length : ℝ := by sorry

theorem rectangle_composition (total_width total_height : ℝ) 
  (h_width : total_width = 3500)
  (h_height : total_height = 2100)
  (h_composition : ∃ (r : ℝ), 2 * r + square_side_length = total_height ∧ 
                               (square_side_length + 100) + square_side_length + (square_side_length + 200) = total_width) :
  square_side_length = 1066.67 := by sorry

end NUMINAMATH_CALUDE_rectangle_composition_l2695_269578


namespace NUMINAMATH_CALUDE_min_blocks_for_slotted_structure_l2695_269548

/-- A block with one hook and five slots -/
structure Block :=
  (hook : Fin 6)
  (slots : Finset (Fin 6))
  (hook_slot_distinct : hook ∉ slots)
  (slot_count : slots.card = 5)

/-- A structure made of blocks -/
structure Structure :=
  (blocks : Finset Block)
  (no_visible_hooks : ∀ b ∈ blocks, ∃ b' ∈ blocks, b.hook ∈ b'.slots)

/-- The theorem stating that the minimum number of blocks required is 4 -/
theorem min_blocks_for_slotted_structure :
  ∀ s : Structure, s.blocks.card ≥ 4 ∧ 
  ∃ s' : Structure, s'.blocks.card = 4 :=
sorry

end NUMINAMATH_CALUDE_min_blocks_for_slotted_structure_l2695_269548


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2695_269527

/-- The line of reflection for a point (x₁, y₁) to (x₂, y₂) has slope m and y-intercept b -/
def is_reflection_line (x₁ y₁ x₂ y₂ m b : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  (midpoint_y = m * midpoint_x + b) ∧ 
  (m * ((x₂ - x₁) / 2) = (y₁ - y₂) / 2)

/-- The sum of slope and y-intercept of the reflection line for (2, 3) to (10, 7) is 3 -/
theorem reflection_line_sum :
  ∃ (m b : ℝ), is_reflection_line 2 3 10 7 m b ∧ m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2695_269527


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2695_269516

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 500) 
  (h2 : bridge_length = 350) 
  (h3 : crossing_time = 60) : 
  ∃ (speed : ℝ), abs (speed - 14.1667) < 0.0001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l2695_269516


namespace NUMINAMATH_CALUDE_opposite_of_one_minus_cube_root_three_l2695_269529

theorem opposite_of_one_minus_cube_root_three :
  -(1 - Real.rpow 3 (1/3)) = Real.rpow 3 (1/3) - 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_minus_cube_root_three_l2695_269529


namespace NUMINAMATH_CALUDE_cricket_matches_played_l2695_269562

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  matches_played : ℕ
  total_runs : ℕ

/-- Calculate the batting average of a player -/
def batting_average (player : CricketPlayer) : ℚ :=
  player.total_runs / player.matches_played

theorem cricket_matches_played 
  (rahul ankit : CricketPlayer)
  (h1 : batting_average rahul = 46)
  (h2 : batting_average ankit = 52)
  (h3 : batting_average {matches_played := rahul.matches_played + 1, 
                         total_runs := rahul.total_runs + 78} = 54)
  (h4 : ∃ x : ℕ, 
        batting_average {matches_played := rahul.matches_played + 1, 
                         total_runs := rahul.total_runs + 78} = 54 ∧
        batting_average {matches_played := ankit.matches_played + 1, 
                         total_runs := ankit.total_runs + x} = 54) :
  rahul.matches_played = 3 ∧ ankit.matches_played = 3 := by
sorry

end NUMINAMATH_CALUDE_cricket_matches_played_l2695_269562


namespace NUMINAMATH_CALUDE_binary_11111111_eq_2_pow_8_minus_1_l2695_269524

/-- Converts a binary number represented as a list of bits (0s and 1s) to its decimal equivalent. -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Theorem: The binary number (11111111)₂ is equal to 2^8 - 1 in decimal form. -/
theorem binary_11111111_eq_2_pow_8_minus_1 :
  binary_to_decimal [1, 1, 1, 1, 1, 1, 1, 1] = 2^8 - 1 := by
  sorry

#eval binary_to_decimal [1, 1, 1, 1, 1, 1, 1, 1]

end NUMINAMATH_CALUDE_binary_11111111_eq_2_pow_8_minus_1_l2695_269524


namespace NUMINAMATH_CALUDE_not_proportional_l2695_269507

/-- A function f is directly proportional to x if there exists a constant k such that f x = k * x for all x -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- A function f is inversely proportional to x if there exists a constant k such that f x = k / x for all non-zero x -/
def InverselyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function defined by the equation x^2 + y = 1 -/
def f (x : ℝ) : ℝ := 1 - x^2

theorem not_proportional : ¬(DirectlyProportional f) ∧ ¬(InverselyProportional f) := by
  sorry

end NUMINAMATH_CALUDE_not_proportional_l2695_269507


namespace NUMINAMATH_CALUDE_intersection_point_median_altitude_l2695_269526

/-- Given a triangle ABC with vertices A(5,1), B(-1,-3), and C(4,3),
    the intersection point of the median CM and altitude BN
    has coordinates (5/3, -5/3). -/
theorem intersection_point_median_altitude (A B C M N : ℝ × ℝ) :
  A = (5, 1) →
  B = (-1, -3) →
  C = (4, 3) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  (N.2 - B.2) * (C.1 - A.1) = (C.2 - A.2) * (N.1 - B.1) →
  (∃ t : ℝ, C + t • (M - C) = N) →
  N = (5/3, -5/3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_median_altitude_l2695_269526


namespace NUMINAMATH_CALUDE_train_speed_train_speed_approx_66_l2695_269510

/-- The speed of a train given its length, the time it takes to pass a man running in the opposite direction, and the man's speed. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_ms := relative_speed - man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  train_speed_kmh

/-- The speed of the train is approximately 66 km/h given the specified conditions. -/
theorem train_speed_approx_66 :
  ∃ ε > 0, abs (train_speed 120 6 6 - 66) < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_approx_66_l2695_269510


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l2695_269555

/-- The number of ways to arrange a chess team with specific conditions -/
theorem chess_team_arrangements (num_boys num_girls : ℕ) : 
  num_boys = 3 → num_girls = 2 → (num_boys.factorial * num_girls.factorial) = 12 := by
  sorry

#check chess_team_arrangements

end NUMINAMATH_CALUDE_chess_team_arrangements_l2695_269555


namespace NUMINAMATH_CALUDE_max_length_sum_l2695_269546

/-- Length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer -/
def length (n : ℕ) : ℕ := sorry

theorem max_length_sum (x y : ℕ) (hx : x > 1) (hy : y > 1) (hsum : x + 3 * y < 920) :
  ∃ (a b : ℕ), length x + length y ≤ a + b ∧ a + b = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_length_sum_l2695_269546


namespace NUMINAMATH_CALUDE_triangle_side_length_l2695_269530

/-- An equilateral triangle with a point inside and perpendiculars to its sides. -/
structure TriangleWithPoint where
  /-- Side length of the equilateral triangle -/
  side_length : ℝ
  /-- Distance from the point to side AB -/
  dist_to_AB : ℝ
  /-- Distance from the point to side BC -/
  dist_to_BC : ℝ
  /-- Distance from the point to side CA -/
  dist_to_CA : ℝ
  /-- The triangle is equilateral -/
  equilateral : side_length > 0
  /-- The point is inside the triangle -/
  point_inside : dist_to_AB > 0 ∧ dist_to_BC > 0 ∧ dist_to_CA > 0

/-- Theorem: If the perpendicular distances are 2, 2√2, and 4, then the side length is 4√3 + (4√6)/3 -/
theorem triangle_side_length (t : TriangleWithPoint) 
  (h1 : t.dist_to_AB = 2) 
  (h2 : t.dist_to_BC = 2 * Real.sqrt 2) 
  (h3 : t.dist_to_CA = 4) : 
  t.side_length = 4 * Real.sqrt 3 + (4 * Real.sqrt 6) / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2695_269530


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2695_269567

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A square inscribed in the region bounded by the parabola and the x-axis -/
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  side : ℝ    -- length of the square's side
  h1 : center - side/2 ≥ 0  -- left side of square is non-negative
  h2 : center + side/2 ≤ 10 -- right side of square is at most the x-intercept
  h3 : parabola (center + side/2) = side -- top-right corner lies on the parabola

theorem inscribed_square_area :
  ∃ (s : InscribedSquare), s.side^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2695_269567


namespace NUMINAMATH_CALUDE_pension_formula_l2695_269566

/-- Represents the annual pension function based on years of service -/
def annual_pension (k : ℝ) (x : ℝ) : ℝ := k * x^2

/-- The pension increase after 4 additional years of service -/
def increase_4_years (k : ℝ) (x : ℝ) : ℝ := annual_pension k (x + 4) - annual_pension k x

/-- The pension increase after 9 additional years of service -/
def increase_9_years (k : ℝ) (x : ℝ) : ℝ := annual_pension k (x + 9) - annual_pension k x

theorem pension_formula (k : ℝ) (x : ℝ) :
  (increase_4_years k x = 144) ∧ 
  (increase_9_years k x = 324) →
  annual_pension k x = (Real.sqrt 171 / 5) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_pension_formula_l2695_269566
