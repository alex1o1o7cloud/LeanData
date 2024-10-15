import Mathlib

namespace NUMINAMATH_CALUDE_handshake_theorem_l2032_203285

/-- The number of handshakes for each student in a class where every two students shake hands once. -/
def handshakes_per_student (n : ℕ) : ℕ := n - 1

/-- The total number of handshakes in a class where every two students shake hands once. -/
def total_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a class of 57 students, if every two students shake hands with each other once, 
    then each student shakes hands 56 times, and the total number of handshakes is (57 × 56) / 2. -/
theorem handshake_theorem :
  handshakes_per_student 57 = 56 ∧ total_handshakes 57 = (57 * 56) / 2 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l2032_203285


namespace NUMINAMATH_CALUDE_rebecca_earnings_l2032_203232

/-- Rebecca's hair salon earnings calculation -/
theorem rebecca_earnings (haircut_price perm_price dye_job_price dye_cost : ℕ)
  (num_haircuts num_perms num_dye_jobs tips : ℕ) :
  haircut_price = 30 →
  perm_price = 40 →
  dye_job_price = 60 →
  dye_cost = 10 →
  num_haircuts = 4 →
  num_perms = 1 →
  num_dye_jobs = 2 →
  tips = 50 →
  (haircut_price * num_haircuts + 
   perm_price * num_perms + 
   dye_job_price * num_dye_jobs + 
   tips - 
   dye_cost * num_dye_jobs) = 310 :=
by sorry

end NUMINAMATH_CALUDE_rebecca_earnings_l2032_203232


namespace NUMINAMATH_CALUDE_larger_number_proof_l2032_203231

theorem larger_number_proof (x y : ℕ) (h1 : x > y) (h2 : x + y = 363) (h3 : x = 16 * y + 6) : x = 342 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2032_203231


namespace NUMINAMATH_CALUDE_fraction_equality_l2032_203278

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2032_203278


namespace NUMINAMATH_CALUDE_chord_of_ellipse_l2032_203244

-- Define the real numbers m, n, s, t
variable (m n s t : ℝ)

-- Define the conditions
def conditions (m n s t : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0 ∧
  m + n = 3 ∧
  m / s + n / t = 1 ∧
  m < n ∧
  ∀ (s' t' : ℝ), s' > 0 → t' > 0 → m / s' + n / t' = 1 → s + t ≤ s' + t'

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 16 = 1

-- Define the chord equation
def chord_equation (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Theorem statement
theorem chord_of_ellipse (m n s t : ℝ) :
  conditions m n s t →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    (x₁ + x₂) / 2 = m ∧ (y₁ + y₂) / 2 = n ∧
    ∀ (x y : ℝ), x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2 → chord_equation x y) :=
by sorry

end NUMINAMATH_CALUDE_chord_of_ellipse_l2032_203244


namespace NUMINAMATH_CALUDE_shelter_puppies_count_l2032_203227

theorem shelter_puppies_count :
  ∀ (puppies kittens : ℕ),
    kittens = 2 * puppies + 14 →
    puppies > 0 →
    kittens = 78 →
    puppies = 32 := by
  sorry

end NUMINAMATH_CALUDE_shelter_puppies_count_l2032_203227


namespace NUMINAMATH_CALUDE_line_parametrization_l2032_203256

/-- The line equation --/
def line_equation (x y : ℝ) : Prop := y = (2/3) * x + 3

/-- The parametric equation of the line --/
def parametric_equation (x y s l t : ℝ) : Prop :=
  x = -9 + t * l ∧ y = s + t * (-7)

/-- The theorem stating the values of s and l --/
theorem line_parametrization :
  ∃ (s l : ℝ), (∀ (x y t : ℝ), line_equation x y ↔ parametric_equation x y s l t) ∧ s = -3 ∧ l = -10.5 := by
  sorry

end NUMINAMATH_CALUDE_line_parametrization_l2032_203256


namespace NUMINAMATH_CALUDE_solve_sticker_problem_l2032_203229

def sticker_problem (initial : ℝ) (bought : ℝ) (birthday : ℝ) (mother : ℝ) (total : ℝ) : Prop :=
  let from_sister := total - (initial + bought + birthday + mother)
  from_sister = 6.0

theorem solve_sticker_problem :
  sticker_problem 20.0 26.0 20.0 58.0 130.0 := by
  sorry

end NUMINAMATH_CALUDE_solve_sticker_problem_l2032_203229


namespace NUMINAMATH_CALUDE_smallest_congruent_difference_l2032_203205

theorem smallest_congruent_difference : ∃ m n : ℕ,
  (m ≥ 100 ∧ m < 1000 ∧ m % 13 = 3 ∧ ∀ k, k ≥ 100 ∧ k < 1000 ∧ k % 13 = 3 → m ≤ k) ∧
  (n ≥ 1000 ∧ n < 10000 ∧ n % 13 = 3 ∧ ∀ l, l ≥ 1000 ∧ l < 10000 ∧ l % 13 = 3 → n ≤ l) ∧
  n - m = 896 :=
by sorry

end NUMINAMATH_CALUDE_smallest_congruent_difference_l2032_203205


namespace NUMINAMATH_CALUDE_inequality_holds_iff_even_l2032_203275

theorem inequality_holds_iff_even (n : ℕ+) :
  (∀ x : ℝ, 3 * x^(n : ℕ) + n * (x + 2) - 3 ≥ n * x^2) ↔ Even n := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_even_l2032_203275


namespace NUMINAMATH_CALUDE_parking_lot_cars_l2032_203252

theorem parking_lot_cars (total_wheels : ℕ) (wheels_per_car : ℕ) (h1 : total_wheels = 68) (h2 : wheels_per_car = 4) :
  total_wheels / wheels_per_car = 17 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l2032_203252


namespace NUMINAMATH_CALUDE_tan_22_5_deg_decomposition_l2032_203219

theorem tan_22_5_deg_decomposition :
  ∃ (a b c d : ℕ+),
    (Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - b + (c : ℝ).sqrt - (d : ℝ).sqrt) ∧
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (a + b + c + d = 3) := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_decomposition_l2032_203219


namespace NUMINAMATH_CALUDE_circle_and_point_position_l2032_203298

-- Define the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 18/5)^2 + y^2 = 569/25

-- Define the points
def point_A : ℝ × ℝ := (1, 4)
def point_B : ℝ × ℝ := (3, 2)
def point_P : ℝ × ℝ := (2, 4)

-- Define what it means for a point to be on the circle
def on_circle (p : ℝ × ℝ) : Prop :=
  circle_equation p.1 p.2

-- Define what it means for a point to be inside the circle
def inside_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 18/5)^2 + p.2^2 < 569/25

-- Theorem statement
theorem circle_and_point_position :
  (on_circle point_A) ∧ 
  (on_circle point_B) ∧ 
  (inside_circle point_P) := by
  sorry

end NUMINAMATH_CALUDE_circle_and_point_position_l2032_203298


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2032_203235

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 - 3

-- Define the theorem
theorem quadratic_max_value (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ a → f x ≤ 15) ∧
  (∃ x, 1 ≤ x ∧ x ≤ a ∧ f x = 15) →
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2032_203235


namespace NUMINAMATH_CALUDE_article_cost_l2032_203276

/-- The cost of an article given selling conditions. -/
theorem article_cost (sell_price_1 sell_price_2 : ℚ) (gain_increase : ℚ) : 
  sell_price_1 = 700 →
  sell_price_2 = 750 →
  gain_increase = 1/10 →
  ∃ (cost gain : ℚ), 
    cost + gain = sell_price_1 ∧
    cost + gain * (1 + gain_increase) = sell_price_2 ∧
    cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l2032_203276


namespace NUMINAMATH_CALUDE_twelve_customers_in_line_l2032_203223

/-- The number of customers in a restaurant line -/
def customers_in_line (people_behind_front : ℕ) : ℕ :=
  people_behind_front + 1

/-- Theorem: Given 11 people behind the front person, there are 12 customers in line -/
theorem twelve_customers_in_line :
  customers_in_line 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_customers_in_line_l2032_203223


namespace NUMINAMATH_CALUDE_increasing_sequence_with_properties_l2032_203247

theorem increasing_sequence_with_properties :
  ∃ (a : ℕ → ℕ) (C : ℝ), 
    (∀ n, a n < a (n + 1)) ∧ 
    (∀ m : ℕ+, ∃! (i j : ℕ), m = a j - a i) ∧
    (∀ k : ℕ+, (a k : ℝ) ≤ C * (k : ℝ)^3) :=
sorry

end NUMINAMATH_CALUDE_increasing_sequence_with_properties_l2032_203247


namespace NUMINAMATH_CALUDE_power_sum_equality_l2032_203284

theorem power_sum_equality : 3^(3+4+5) + (3^3 + 3^4 + 3^5) = 531792 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2032_203284


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l2032_203246

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l2032_203246


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2032_203297

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ n : ℤ, (d * n^3 + c * n^2 + b * n + a) % 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2032_203297


namespace NUMINAMATH_CALUDE_new_person_weight_l2032_203202

/-- Given a group of 8 people, if replacing one person weighing 65 kg with a new person
    increases the average weight by 3.5 kg, then the weight of the new person is 93 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 93 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2032_203202


namespace NUMINAMATH_CALUDE_tangent_circle_to_sphere_reasoning_l2032_203288

/-- Represents the type of reasoning used in geometric analogies --/
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical
  | Other

/-- Represents a geometric property in 2D --/
structure Property2D where
  statement : String

/-- Represents a geometric property in 3D --/
structure Property3D where
  statement : String

/-- The property of tangent lines to circles in 2D --/
def tangentLineCircle : Property2D :=
  { statement := "When a line is tangent to a circle, the line connecting the center of the circle to the tangent point is perpendicular to the line" }

/-- The property of tangent planes to spheres in 3D --/
def tangentPlaneSphere : Property3D :=
  { statement := "When a plane is tangent to a sphere, the line connecting the center of the sphere to the tangent point is perpendicular to the plane" }

/-- The theorem stating that the reasoning used to extend the 2D property to 3D is analogical --/
theorem tangent_circle_to_sphere_reasoning :
  (∃ (p2d : Property2D) (p3d : Property3D), p2d = tangentLineCircle ∧ p3d = tangentPlaneSphere) →
  (∃ (r : ReasoningType), r = ReasoningType.Analogical) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_to_sphere_reasoning_l2032_203288


namespace NUMINAMATH_CALUDE_zyx_syndrome_ratio_is_one_to_three_l2032_203214

/-- Represents the ratio of patients with ZYX syndrome to those without it -/
structure ZYXRatio where
  with_syndrome : ℕ
  without_syndrome : ℕ

/-- The clinic's patient information -/
structure ClinicInfo where
  total_patients : ℕ
  diagnosed_patients : ℕ

/-- Calculates the ZYX syndrome ratio given clinic information -/
def calculate_zyx_ratio (info : ClinicInfo) : ZYXRatio :=
  { with_syndrome := info.diagnosed_patients,
    without_syndrome := info.total_patients - info.diagnosed_patients }

/-- Simplifies a ZYX ratio by dividing both numbers by their GCD -/
def simplify_ratio (ratio : ZYXRatio) : ZYXRatio :=
  let gcd := Nat.gcd ratio.with_syndrome ratio.without_syndrome
  { with_syndrome := ratio.with_syndrome / gcd,
    without_syndrome := ratio.without_syndrome / gcd }

theorem zyx_syndrome_ratio_is_one_to_three :
  let clinic_info : ClinicInfo := { total_patients := 52, diagnosed_patients := 13 }
  let ratio := simplify_ratio (calculate_zyx_ratio clinic_info)
  ratio.with_syndrome = 1 ∧ ratio.without_syndrome = 3 := by sorry

end NUMINAMATH_CALUDE_zyx_syndrome_ratio_is_one_to_three_l2032_203214


namespace NUMINAMATH_CALUDE_sugar_content_per_bar_l2032_203259

/-- The sugar content of each chocolate bar -/
def sugar_per_bar (total_sugar total_bars lollipop_sugar : ℕ) : ℚ :=
  (total_sugar - lollipop_sugar) / total_bars

/-- Proof that the sugar content of each chocolate bar is 10 grams -/
theorem sugar_content_per_bar :
  sugar_per_bar 177 14 37 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_content_per_bar_l2032_203259


namespace NUMINAMATH_CALUDE_evaluate_expression_l2032_203200

theorem evaluate_expression : 4 - (-3)^(-1/2 : ℂ) = 4 + (Complex.I * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2032_203200


namespace NUMINAMATH_CALUDE_card_drawing_certainty_l2032_203230

theorem card_drawing_certainty (total : ℕ) (hearts clubs spades drawn : ℕ) 
  (h_total : total = hearts + clubs + spades)
  (h_hearts : hearts = 5)
  (h_clubs : clubs = 4)
  (h_spades : spades = 3)
  (h_drawn : drawn = 10) :
  ∀ (draw : Finset ℕ), draw.card = drawn → 
    (∃ (h c s : ℕ), h ∈ draw ∧ c ∈ draw ∧ s ∈ draw ∧ 
      h ≤ hearts ∧ c ≤ clubs ∧ s ≤ spades) :=
sorry

end NUMINAMATH_CALUDE_card_drawing_certainty_l2032_203230


namespace NUMINAMATH_CALUDE_remaining_money_l2032_203280

def initial_amount : ℕ := 760
def ticket_price : ℕ := 300
def hotel_price : ℕ := ticket_price / 2

theorem remaining_money :
  initial_amount - (ticket_price + hotel_price) = 310 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l2032_203280


namespace NUMINAMATH_CALUDE_total_spent_l2032_203291

def trick_deck_price : ℕ := 8
def victor_decks : ℕ := 6
def friend_decks : ℕ := 2

theorem total_spent : 
  trick_deck_price * victor_decks + trick_deck_price * friend_decks = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_l2032_203291


namespace NUMINAMATH_CALUDE_gcd_lcm_product_240_l2032_203210

theorem gcd_lcm_product_240 : 
  ∃! (s : Finset Nat), 
    (∀ d ∈ s, ∃ a b : Nat, a > 0 ∧ b > 0 ∧ Nat.gcd a b = d ∧ Nat.gcd a b * Nat.lcm a b = 240) ∧ 
    (∀ d : Nat, (∃ a b : Nat, a > 0 ∧ b > 0 ∧ Nat.gcd a b = d ∧ Nat.gcd a b * Nat.lcm a b = 240) → d ∈ s) ∧
    s.card = 10 :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_240_l2032_203210


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2032_203245

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2032_203245


namespace NUMINAMATH_CALUDE_measure_gold_dust_l2032_203206

/-- Represents the available weights for measuring gold dust -/
inductive Weight
  | TwoHundredGram
  | FiftyGram

/-- Represents a case with different available weights -/
inductive Case
  | CaseA
  | CaseB

/-- Represents a weighing operation on a balance scale -/
def Weighing := ℝ → ℝ → Prop

/-- Represents the ability to measure a specific amount of gold dust -/
def CanMeasure (totalGold : ℝ) (targetAmount : ℝ) (weights : List Weight) (case : Case) : Prop :=
  ∃ (w1 w2 w3 : Weighing), 
    (w1 totalGold targetAmount) ∧ 
    (w2 totalGold targetAmount) ∧ 
    (w3 totalGold targetAmount)

/-- The main theorem stating that it's possible to measure 2 kg of gold dust in both cases -/
theorem measure_gold_dust : 
  ∀ (case : Case),
    CanMeasure 9 2 
      (match case with
        | Case.CaseA => [Weight.TwoHundredGram, Weight.FiftyGram]
        | Case.CaseB => [Weight.TwoHundredGram])
      case :=
by
  sorry

end NUMINAMATH_CALUDE_measure_gold_dust_l2032_203206


namespace NUMINAMATH_CALUDE_number_equation_l2032_203257

theorem number_equation (x : ℝ) : 150 - x = x + 68 ↔ x = 41 := by sorry

end NUMINAMATH_CALUDE_number_equation_l2032_203257


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2032_203207

/-- Given two lines l₁ and l₂ with equations 3x + 2y - 2 = 0 and (2m-1)x + my + 1 = 0 respectively,
    if l₁ is parallel to l₂, then m = 2. -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, 3 * x + 2 * y - 2 = 0 ↔ (2 * m - 1) * x + m * y + 1 = 0) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2032_203207


namespace NUMINAMATH_CALUDE_student_rabbit_difference_is_95_l2032_203236

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 22

/-- The number of rabbits in each fourth-grade classroom -/
def rabbits_per_classroom : ℕ := 3

/-- The number of fourth-grade classrooms -/
def number_of_classrooms : ℕ := 5

/-- The difference between the total number of students and rabbits in all classrooms -/
def student_rabbit_difference : ℕ :=
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms)

theorem student_rabbit_difference_is_95 :
  student_rabbit_difference = 95 := by sorry

end NUMINAMATH_CALUDE_student_rabbit_difference_is_95_l2032_203236


namespace NUMINAMATH_CALUDE_age_difference_l2032_203218

theorem age_difference (A B C : ℕ) (h1 : C = A - 16) : A + B - (B + C) = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2032_203218


namespace NUMINAMATH_CALUDE_intersection_M_N_l2032_203212

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2032_203212


namespace NUMINAMATH_CALUDE_trig_identities_l2032_203251

/-- Prove trigonometric identities -/
theorem trig_identities :
  let π : ℝ := Real.pi
  let cos_45 : ℝ := Real.cos (π / 4)
  let tan_30 : ℝ := Real.tan (π / 6)
  let cos_30 : ℝ := Real.cos (π / 6)
  let sin_60 : ℝ := Real.sin (π / 3)
  let sin_30 : ℝ := Real.sin (π / 6)
  let tan_60 : ℝ := Real.tan (π / 3)
  2 * cos_45 - (3 / 2) * tan_30 * cos_30 + sin_60 ^ 2 = Real.sqrt 2 ∧
  (sin_30)⁻¹ * (sin_60 - cos_45) - Real.sqrt ((1 - tan_60) ^ 2) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l2032_203251


namespace NUMINAMATH_CALUDE_solution_set_implies_a_b_values_f_1_negative_implies_a_conditions_l2032_203225

-- Define the function f
def f (a b x : ℝ) : ℝ := -3 * x^2 + a * (5 - a) * x + b

-- Part 1
theorem solution_set_implies_a_b_values (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) :=
sorry

-- Part 2
theorem f_1_negative_implies_a_conditions (b : ℝ) :
  (∀ a : ℝ, f a b 1 < 0 ↔
    (b < -13/4 ∧ a ∈ Set.univ) ∨
    (b = -13/4 ∧ a ≠ 5/2) ∨
    (b > -13/4 ∧ (a > (5 + Real.sqrt (4*b + 13))/2 ∨ a < (5 - Real.sqrt (4*b + 13))/2))) :=
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_b_values_f_1_negative_implies_a_conditions_l2032_203225


namespace NUMINAMATH_CALUDE_negation_of_exists_cube_positive_l2032_203234

theorem negation_of_exists_cube_positive :
  (¬ ∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_cube_positive_l2032_203234


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2032_203237

theorem negative_fraction_comparison : -1/3 < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2032_203237


namespace NUMINAMATH_CALUDE_inequality_proof_l2032_203209

theorem inequality_proof (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a * b + b * c + c * a = 1) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2032_203209


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2032_203233

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    its eccentricity is 2, and the distance from the origin to line AB
    (where A(a, 0) and B(0, -b)) is 3/2, prove that the equation of the
    hyperbola is x²/3 - y²/9 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c / a = 2) →
  (∃ d : ℝ, d = 3/2 ∧ d = |(a * b)| / Real.sqrt (a^2 + b^2)) →
  (∀ x y : ℝ, x^2 / 3 - y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2032_203233


namespace NUMINAMATH_CALUDE_total_triangles_is_sixteen_l2032_203224

/-- Represents the count of triangles in each size category -/
structure TriangleCounts where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of triangles -/
def totalTriangles (counts : TriangleCounts) : Nat :=
  counts.small + counts.medium + counts.large

/-- The given triangle counts for the figure -/
def figureCounts : TriangleCounts :=
  { small := 11, medium := 4, large := 1 }

/-- Theorem stating that the total number of triangles in the figure is 16 -/
theorem total_triangles_is_sixteen :
  totalTriangles figureCounts = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_sixteen_l2032_203224


namespace NUMINAMATH_CALUDE_f_triple_3_l2032_203271

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_triple_3 : f (f (f 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_triple_3_l2032_203271


namespace NUMINAMATH_CALUDE_lauren_jane_equation_l2032_203274

theorem lauren_jane_equation (x : ℝ) :
  (∀ x, |x - 4| = 3 ↔ x^2 + b*x + c = 0) →
  (b : ℝ) = -8 ∧ (c : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_lauren_jane_equation_l2032_203274


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2032_203270

/-- A point is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- For any real number x, the point (x^2 + 1, -4) is in the fourth quadrant -/
theorem point_in_fourth_quadrant (x : ℝ) :
  in_fourth_quadrant (x^2 + 1, -4) := by
  sorry


end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2032_203270


namespace NUMINAMATH_CALUDE_min_xy_and_x_plus_y_l2032_203239

/-- Given positive real numbers x and y satisfying x + 8y - xy = 0,
    proves that the minimum value of xy is 32 and
    the minimum value of x + y is 9 + 4√2 -/
theorem min_xy_and_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8*y - x*y = 0) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 8*y' - x'*y' = 0 → x*y ≤ x'*y') ∧
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 8*y' - x'*y' = 0 → x + y ≤ x' + y') ∧
  x*y = 32 ∧ x + y = 9 + 4*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_and_x_plus_y_l2032_203239


namespace NUMINAMATH_CALUDE_farmer_cows_problem_l2032_203294

theorem farmer_cows_problem (initial_food : ℝ) (initial_cows : ℕ) :
  initial_food > 0 →
  initial_cows > 0 →
  (initial_food / 50 = initial_food / (5 * 10)) →
  (initial_cows = 200) :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_problem_l2032_203294


namespace NUMINAMATH_CALUDE_toy_car_production_l2032_203222

theorem toy_car_production (yesterday : ℕ) (today : ℕ) : 
  yesterday = 60 → today = 2 * yesterday → yesterday + today = 180 := by
  sorry

end NUMINAMATH_CALUDE_toy_car_production_l2032_203222


namespace NUMINAMATH_CALUDE_line_equation_of_l_l2032_203243

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The line l passing through (3,2) with slope -4 -/
def l : Line := { point := (3, 2), slope := -4 }

/-- Theorem: The equation of line l is 4x + y - 14 = 0 -/
theorem line_equation_of_l : 
  ∃ (eq : LineEquation), eq.a = 4 ∧ eq.b = 1 ∧ eq.c = -14 ∧
  ∀ (x y : ℝ), eq.a * x + eq.b * y + eq.c = 0 ↔ y - l.point.2 = l.slope * (x - l.point.1) :=
sorry

end NUMINAMATH_CALUDE_line_equation_of_l_l2032_203243


namespace NUMINAMATH_CALUDE_ellipse_properties_l2032_203228

-- Define the ellipse C
def ellipse_C (x y a b c : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0 ∧ a = 2*c

-- Define the circle P1
def circle_P1 (x y r : ℝ) : Prop :=
  (x + 4*Real.sqrt 3 / 7)^2 + (y - 3*Real.sqrt 3 / 7)^2 = r^2 ∧ r > 0

-- Define the theorem
theorem ellipse_properties :
  ∀ (a b c : ℝ),
  ellipse_C (Real.sqrt 3) ((Real.sqrt 3) / 2) a b c →
  ellipse_C (-a + 2*c) 0 a b c →
  (∃ (x y r : ℝ), circle_P1 x y r ∧ ellipse_C x y a b c) →
  (∃ (k : ℝ), k > 1 ∧
    (∀ (x y : ℝ), y = k*(x + 1) → 
      (∃ (p q : ℝ), ellipse_C p (k*(p + 1)) a b c ∧ 
                    ellipse_C q (k*(q + 1)) a b c ∧
                    9/4 < (1 + k^2) * (9 / (3 + 4*k^2)) ∧
                    (1 + k^2) * (9 / (3 + 4*k^2)) ≤ 12/5))) →
  c / a = 1/2 ∧ a = 2 ∧ b = Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_ellipse_properties_l2032_203228


namespace NUMINAMATH_CALUDE_room_length_is_correct_l2032_203253

/-- The length of a rectangular room -/
def room_length : ℝ := 5.5

/-- The width of the room -/
def room_width : ℝ := 3.75

/-- The cost of paving the floor -/
def paving_cost : ℝ := 12375

/-- The rate of paving per square meter -/
def paving_rate : ℝ := 600

/-- Theorem stating that the room length is correct given the conditions -/
theorem room_length_is_correct : 
  room_length * room_width * paving_rate = paving_cost := by sorry

end NUMINAMATH_CALUDE_room_length_is_correct_l2032_203253


namespace NUMINAMATH_CALUDE_statistics_collection_count_l2032_203299

/-- Represents the multiset of letters in "STATISTICS" --/
def statistics : Multiset Char := {'S', 'T', 'A', 'T', 'I', 'S', 'T', 'I', 'C', 'S'}

/-- Represents the vowels in "STATISTICS" --/
def vowels : Multiset Char := {'A', 'I', 'I'}

/-- Represents the consonants in "STATISTICS", with S and T treated as indistinguishable --/
def consonants : Multiset Char := {'C', 'S', 'S', 'S'}

/-- The number of distinct collections of 7 letters (3 vowels and 4 consonants) from "STATISTICS" --/
def distinct_collections : ℕ := 30

theorem statistics_collection_count :
  (Multiset.card statistics = 10) →
  (Multiset.card vowels = 3) →
  (Multiset.card consonants = 4) →
  (∀ x ∈ vowels, x ∈ statistics) →
  (∀ x ∈ consonants, x ∈ statistics ∨ x = 'S') →
  (distinct_collections = 30) := by
  sorry

end NUMINAMATH_CALUDE_statistics_collection_count_l2032_203299


namespace NUMINAMATH_CALUDE_roots_of_equation_l2032_203283

theorem roots_of_equation (x : ℝ) : (x + 1)^2 = 0 ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2032_203283


namespace NUMINAMATH_CALUDE_proposition_logic_l2032_203263

theorem proposition_logic (p q : Prop) (hp : p ↔ (3 + 3 = 5)) (hq : q ↔ (6 > 3)) :
  (p ∨ q) ∧ (¬q ↔ False) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_l2032_203263


namespace NUMINAMATH_CALUDE_work_completion_time_l2032_203203

/-- Given workers a and b, where:
    - a can complete the work in 20 days
    - a and b together can complete the work in 15 days when b works half-time
    Prove that a and b together can complete the work in 12 days when b works full-time -/
theorem work_completion_time (a b : ℝ) 
  (ha : a = 1 / 20)  -- a's work rate per day
  (hab_half : a + b / 2 = 1 / 15)  -- combined work rate when b works half-time
  : a + b = 1 / 12 := by  -- combined work rate when b works full-time
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2032_203203


namespace NUMINAMATH_CALUDE_total_investment_l2032_203260

theorem total_investment (T : ℝ) : T = 2000 :=
  let invested_at_8_percent : ℝ := 600
  let invested_at_10_percent : ℝ := T - 600
  let income_difference : ℝ := 92
  have h1 : 0.10 * invested_at_10_percent - 0.08 * invested_at_8_percent = income_difference := by sorry
  sorry

end NUMINAMATH_CALUDE_total_investment_l2032_203260


namespace NUMINAMATH_CALUDE_mixture_replacement_theorem_l2032_203295

/-- The amount of mixture replaced to change the ratio from 7:5 to 7:9 -/
def mixture_replaced (initial_total : ℝ) (replaced : ℝ) : Prop :=
  let initial_a := 21
  let initial_b := initial_total - initial_a
  let new_b := initial_b + replaced
  (initial_a / initial_total = 7 / 12) ∧
  (initial_a / new_b = 7 / 9) ∧
  replaced = 12

/-- Theorem stating that 12 liters of mixture were replaced -/
theorem mixture_replacement_theorem :
  ∃ (initial_total : ℝ), mixture_replaced initial_total 12 := by
  sorry

end NUMINAMATH_CALUDE_mixture_replacement_theorem_l2032_203295


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2032_203249

/-- A shape formed by unit cubes in a straight line -/
structure LineShape where
  num_cubes : ℕ

/-- Volume of a LineShape -/
def volume (shape : LineShape) : ℕ :=
  shape.num_cubes

/-- Surface area of a LineShape -/
def surface_area (shape : LineShape) : ℕ :=
  2 * 5 + (shape.num_cubes - 2) * 4

/-- Theorem stating the ratio of volume to surface area for a LineShape with 8 cubes -/
theorem volume_to_surface_area_ratio (shape : LineShape) (h : shape.num_cubes = 8) :
  (volume shape : ℚ) / (surface_area shape : ℚ) = 4 / 17 := by
  sorry


end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2032_203249


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_cubed_l2032_203272

theorem sqrt_fifth_power_cubed : (Real.sqrt ((Real.sqrt 5) ^ 4)) ^ 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_cubed_l2032_203272


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l2032_203240

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for lines and planes
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the property of a line being outside a plane
variable (outside : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes_from_perpendicular_lines
  (α β : Plane) (m n : Line)
  (different_planes : α ≠ β)
  (distinct_lines : m ≠ n)
  (m_outside_α : outside m α)
  (m_outside_β : outside m β)
  (n_outside_α : outside n α)
  (n_outside_β : outside n β)
  (h1 : perpendicular_lines m n)
  (h3 : perpendicular_line_plane n β)
  (h4 : perpendicular_line_plane m α) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_perpendicular_lines_l2032_203240


namespace NUMINAMATH_CALUDE_factorial_ratio_l2032_203281

theorem factorial_ratio : (50 : ℕ).factorial / (48 : ℕ).factorial = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2032_203281


namespace NUMINAMATH_CALUDE_square_ge_of_ge_pos_l2032_203289

theorem square_ge_of_ge_pos {a b : ℝ} (h1 : a ≥ b) (h2 : b > 0) : a^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_ge_of_ge_pos_l2032_203289


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2032_203204

/-- Given a line L1 with equation x - 3y + 2 = 0 and a point P(1, 2),
    prove that the line L2 with equation 3x + y - 5 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 3*y + 2 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 3*x + y - 5 = 0
  let P : ℝ × ℝ := (1, 2)
  (L2 P.1 P.2) ∧                        -- L2 passes through P
  (∀ x1 y1 x2 y2 : ℝ,                   -- L2 is perpendicular to L1
    L1 x1 y1 → L1 x2 y2 → L2 x1 y1 → L2 x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    ((x2 - x1) * 1 + (y2 - y1) * (-3)) * ((x2 - x1) * 3 + (y2 - y1) * 1) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2032_203204


namespace NUMINAMATH_CALUDE_square_of_binomial_l2032_203221

theorem square_of_binomial (m : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, x^2 - 12*x + m = (x + c)^2) → m = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2032_203221


namespace NUMINAMATH_CALUDE_range_of_a_for_solution_a_value_for_minimum_l2032_203264

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Theorem for part I
theorem range_of_a_for_solution (a : ℝ) :
  (∃ x, f a x ≤ 2 - |x - 1|) ↔ (0 ≤ a ∧ a ≤ 4) :=
sorry

-- Theorem for part II
theorem a_value_for_minimum (a : ℝ) :
  a < 2 → (∀ x, f a x ≥ 3) → (∃ x, f a x = 3) → a = -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_solution_a_value_for_minimum_l2032_203264


namespace NUMINAMATH_CALUDE_henrys_earnings_per_lawn_l2032_203266

theorem henrys_earnings_per_lawn 
  (total_lawns : ℕ) 
  (unmowed_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 12) 
  (h2 : unmowed_lawns = 7) 
  (h3 : total_earnings = 25) : 
  total_earnings / (total_lawns - unmowed_lawns) = 5 := by
  sorry

end NUMINAMATH_CALUDE_henrys_earnings_per_lawn_l2032_203266


namespace NUMINAMATH_CALUDE_complex_division_result_l2032_203215

theorem complex_division_result : (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l2032_203215


namespace NUMINAMATH_CALUDE_pool_filling_time_l2032_203273

-- Define the rates of the valves
variable (a b c d : ℝ)

-- Define the conditions
def condition1 : Prop := 1/a + 1/b + 1/c = 1/12
def condition2 : Prop := 1/b + 1/c + 1/d = 1/15
def condition3 : Prop := 1/a + 1/d = 1/20

-- Theorem statement
theorem pool_filling_time 
  (h1 : condition1 a b c)
  (h2 : condition2 b c d)
  (h3 : condition3 a d) :
  1/a + 1/b + 1/c + 1/d = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_pool_filling_time_l2032_203273


namespace NUMINAMATH_CALUDE_platform_length_l2032_203262

/-- The length of a platform given train crossing times -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 420 ∧ platform_time = 60 ∧ pole_time = 30 →
  (train_length / pole_time) * platform_time - train_length = 420 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l2032_203262


namespace NUMINAMATH_CALUDE_range_of_a_l2032_203213

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |4*x - 3| ≤ 1 → x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0) ∧ 
  (∃ x : ℝ, |4*x - 3| ≤ 1 ∧ x^2 - (2*a + 1)*x + a*(a + 1) > 0) → 
  0 ≤ a ∧ a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2032_203213


namespace NUMINAMATH_CALUDE_worker_c_completion_time_l2032_203201

/-- Given workers a, b, and c, and their work rates, prove that c can finish the work in 18 days -/
theorem worker_c_completion_time 
  (total_work : ℝ) 
  (work_rate_a : ℝ) 
  (work_rate_b : ℝ) 
  (work_rate_c : ℝ) 
  (h1 : work_rate_a + work_rate_b + work_rate_c = total_work / 4)
  (h2 : work_rate_a = total_work / 12)
  (h3 : work_rate_b = total_work / 9) :
  work_rate_c = total_work / 18 := by
sorry


end NUMINAMATH_CALUDE_worker_c_completion_time_l2032_203201


namespace NUMINAMATH_CALUDE_right_triangle_similarity_x_values_l2032_203282

theorem right_triangle_similarity_x_values :
  let segments : Finset ℝ := {1, 9, 5, x}
  ∃ (AB CD : ℝ) (a b c d : ℝ),
    AB ∈ segments ∧ CD ∈ segments ∧
    a ∈ segments ∧ b ∈ segments ∧ c ∈ segments ∧ d ∈ segments ∧
    a^2 + b^2 = AB^2 ∧ c^2 + d^2 = CD^2 ∧
    a / c = b / d ∧
    x > 0 →
    (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ y ∈ s, ∃ (AB CD a b c d : ℝ),
      AB ∈ segments ∧ CD ∈ segments ∧
      a ∈ segments ∧ b ∈ segments ∧ c ∈ segments ∧ d ∈ segments ∧
      a^2 + b^2 = AB^2 ∧ c^2 + d^2 = CD^2 ∧
      a / c = b / d ∧
      y = x) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_similarity_x_values_l2032_203282


namespace NUMINAMATH_CALUDE_box_surface_area_and_volume_l2032_203216

/-- Represents the dimensions of a rectangular sheet and the size of square corners to be removed --/
structure BoxParameters where
  length : ℕ
  width : ℕ
  corner_size : ℕ

/-- Calculates the surface area of the interior of the box --/
def calculate_surface_area (params : BoxParameters) : ℕ :=
  params.length * params.width - 4 * params.corner_size * params.corner_size

/-- Calculates the volume of the box --/
def calculate_volume (params : BoxParameters) : ℕ :=
  (params.length - 2 * params.corner_size) * (params.width - 2 * params.corner_size) * params.corner_size

/-- Theorem stating the surface area and volume of the box --/
theorem box_surface_area_and_volume :
  let params : BoxParameters := { length := 25, width := 35, corner_size := 6 }
  calculate_surface_area params = 731 ∧ calculate_volume params = 1794 :=
by sorry

end NUMINAMATH_CALUDE_box_surface_area_and_volume_l2032_203216


namespace NUMINAMATH_CALUDE_triangle_8_8_15_l2032_203220

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is greater than
    the length of the remaining side. -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of line segments with lengths 8cm, 8cm, and 15cm can form a triangle. -/
theorem triangle_8_8_15 : canFormTriangle 8 8 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_8_8_15_l2032_203220


namespace NUMINAMATH_CALUDE_coin_game_probabilities_l2032_203265

-- Define the coin probabilities
def p_heads : ℚ := 3/4
def p_tails : ℚ := 1/4

-- Define the games
def game_A : ℕ := 3  -- number of tosses in Game A
def game_C : ℕ := 4  -- number of tosses in Game C

-- Define the winning probability functions
def win_prob (n : ℕ) : ℚ := p_heads^n + p_tails^n

-- Theorem statement
theorem coin_game_probabilities :
  (win_prob game_A = 7/16) ∧ (win_prob game_C = 41/128) :=
sorry

end NUMINAMATH_CALUDE_coin_game_probabilities_l2032_203265


namespace NUMINAMATH_CALUDE_max_log_sum_l2032_203241

theorem max_log_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4 * y = 40) :
  ∃ (max_val : ℝ), max_val = 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + 4 * b = 40 → Real.log a + Real.log b ≤ max_val := by
  sorry

end NUMINAMATH_CALUDE_max_log_sum_l2032_203241


namespace NUMINAMATH_CALUDE_not_subset_iff_exists_not_mem_l2032_203248

theorem not_subset_iff_exists_not_mem {M P : Set α} (hM : M.Nonempty) :
  ¬(M ⊆ P) ↔ ∃ x ∈ M, x ∉ P := by
  sorry

end NUMINAMATH_CALUDE_not_subset_iff_exists_not_mem_l2032_203248


namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_27_l2032_203261

theorem ab_plus_cd_equals_27
  (a b c d : ℝ)
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -1)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 12) :
  a * b + c * d = 27 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_27_l2032_203261


namespace NUMINAMATH_CALUDE_equation_solution_l2032_203296

theorem equation_solution (x y : ℝ) : 
  x / 3 - y / 2 = 1 → y = 2 * x / 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2032_203296


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2032_203242

/-- Given a geometric sequence with first term -2 and sum of first 3 terms -7/2,
    prove that the common ratio is either 1/2 or -3/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a1 : a 1 = -2)
  (h_S3 : (a 0) + (a 1) + (a 2) = -7/2) :
  (a 1) / (a 0) = 1/2 ∨ (a 1) / (a 0) = -3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2032_203242


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2032_203258

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_standard_equation 
  (major_axis_length : ℝ) 
  (eccentricity : ℝ) :
  major_axis_length = 12 →
  eccentricity = 2/3 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    ∀ x y : ℝ, ellipse_equation a b x y ↔ 
      x^2 / 36 + y^2 / 20 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2032_203258


namespace NUMINAMATH_CALUDE_left_translation_exponential_l2032_203293

/-- Given a function f: ℝ → ℝ, we say it's a left translation by 2 units of g 
    if f(x) = g(x + 2) for all x ∈ ℝ -/
def is_left_translation_by_two (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (x + 2)

/-- The theorem stating that if f is a left translation by 2 units of the function
    x ↦ 2^(2x-1), then f(x) = 2^(2x-5) for all x ∈ ℝ -/
theorem left_translation_exponential 
  (f : ℝ → ℝ) 
  (h : is_left_translation_by_two f (fun x ↦ 2^(2*x - 1))) :
  ∀ x, f x = 2^(2*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_left_translation_exponential_l2032_203293


namespace NUMINAMATH_CALUDE_collinear_vectors_l2032_203208

/-- Given two vectors a and b in R², prove that if 2a + b is collinear with b,
    then the y-coordinate of a is 1/2. -/
theorem collinear_vectors (l x : ℝ) : 
  let a : ℝ × ℝ := (l, x)
  let b : ℝ × ℝ := (4, 2)
  (∃ (k : ℝ), (2 * a.1 + b.1, 2 * a.2 + b.2) = k • b) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l2032_203208


namespace NUMINAMATH_CALUDE_extra_apples_l2032_203292

-- Define the number of red apples
def red_apples : ℕ := 6

-- Define the number of green apples
def green_apples : ℕ := 15

-- Define the number of students who wanted fruit
def students_wanting_fruit : ℕ := 5

-- Define the number of apples each student takes
def apples_per_student : ℕ := 1

-- Theorem to prove
theorem extra_apples : 
  (red_apples + green_apples) - (students_wanting_fruit * apples_per_student) = 16 := by
  sorry

end NUMINAMATH_CALUDE_extra_apples_l2032_203292


namespace NUMINAMATH_CALUDE_ratio_of_first_to_third_term_l2032_203226

/-- An arithmetic sequence with first four terms a, y, b, 3y -/
def ArithmeticSequence (a y b : ℝ) : Prop :=
  ∃ d : ℝ, y - a = d ∧ b - y = d ∧ 3*y - b = d

theorem ratio_of_first_to_third_term (a y b : ℝ) 
  (h : ArithmeticSequence a y b) : a / b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_first_to_third_term_l2032_203226


namespace NUMINAMATH_CALUDE_students_at_start_l2032_203269

theorem students_at_start (students_left : ℕ) (new_students : ℕ) (final_students : ℕ) : 
  students_left = 4 → new_students = 42 → final_students = 48 → 
  final_students - (new_students - students_left) = 10 :=
by sorry

end NUMINAMATH_CALUDE_students_at_start_l2032_203269


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2032_203238

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p^2 + p - 1 = 0) → 
  (q^3 - 2*q^2 + q - 1 = 0) → 
  (r^3 - 2*r^2 + r - 1 = 0) → 
  (1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 20 / 19) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2032_203238


namespace NUMINAMATH_CALUDE_regular_polygon_18_degree_exterior_angles_l2032_203277

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides -/
theorem regular_polygon_18_degree_exterior_angles (n : ℕ) : 
  (n > 0) → (360 / n = 18) → n = 20 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_18_degree_exterior_angles_l2032_203277


namespace NUMINAMATH_CALUDE_sum_divisibility_l2032_203250

theorem sum_divisibility (x y a b S : ℤ) 
  (sum_eq : x + y = S) 
  (masha_div : S ∣ (a * x + b * y)) : 
  S ∣ (b * x + a * y) := by
sorry

end NUMINAMATH_CALUDE_sum_divisibility_l2032_203250


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_at_zero_l2032_203279

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value_at_zero 
  (f : ℝ → ℝ) 
  (hf : MonicQuarticPolynomial f) 
  (h1 : f (-2) = -4)
  (h2 : f 1 = -1)
  (h3 : f 3 = -9)
  (h4 : f 5 = -25) :
  f 0 = -30 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_at_zero_l2032_203279


namespace NUMINAMATH_CALUDE_cone_slant_height_l2032_203217

/-- Given a cone with lateral surface area of 15π when unfolded and base radius of 3, 
    its slant height is 5. -/
theorem cone_slant_height (lateral_area : ℝ) (base_radius : ℝ) : 
  lateral_area = 15 * Real.pi ∧ base_radius = 3 → 
  (lateral_area / (Real.pi * base_radius) : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l2032_203217


namespace NUMINAMATH_CALUDE_marcias_final_hair_length_l2032_203255

/-- Calculates the final hair length after a series of cuts and growth periods --/
def finalHairLength (initialLength : ℝ) 
                    (firstCutPercentage : ℝ) 
                    (firstGrowthMonths : ℕ) 
                    (firstGrowthRate : ℝ) 
                    (secondCutPercentage : ℝ) 
                    (secondGrowthMonths : ℕ) 
                    (secondGrowthRate : ℝ) 
                    (finalCutLength : ℝ) : ℝ :=
  let afterFirstCut := initialLength * (1 - firstCutPercentage)
  let afterFirstGrowth := afterFirstCut + (firstGrowthMonths : ℝ) * firstGrowthRate
  let afterSecondCut := afterFirstGrowth * (1 - secondCutPercentage)
  let afterSecondGrowth := afterSecondCut + (secondGrowthMonths : ℝ) * secondGrowthRate
  afterSecondGrowth - finalCutLength

/-- Theorem stating that Marcia's final hair length is 22.04 inches --/
theorem marcias_final_hair_length : 
  finalHairLength 24 0.3 3 1.5 0.2 5 1.8 4 = 22.04 := by
  sorry

end NUMINAMATH_CALUDE_marcias_final_hair_length_l2032_203255


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2032_203290

theorem quadratic_inequality_solution_set (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2032_203290


namespace NUMINAMATH_CALUDE_average_of_pqrs_l2032_203211

theorem average_of_pqrs (p q r s : ℝ) (h : (8 / 5) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 3.125 := by
sorry

end NUMINAMATH_CALUDE_average_of_pqrs_l2032_203211


namespace NUMINAMATH_CALUDE_cassidy_poster_collection_l2032_203268

/-- The number of posters Cassidy has now -/
def current_posters : ℕ := 22

/-- The number of posters Cassidy will add -/
def added_posters : ℕ := 6

/-- The number of posters Cassidy had two years ago -/
def posters_two_years_ago : ℕ := 14

theorem cassidy_poster_collection :
  2 * posters_two_years_ago = current_posters + added_posters :=
by sorry

end NUMINAMATH_CALUDE_cassidy_poster_collection_l2032_203268


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_power_of_two_l2032_203286

theorem negation_of_existence_squared_greater_than_power_of_two :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_greater_than_power_of_two_l2032_203286


namespace NUMINAMATH_CALUDE_book_price_range_l2032_203254

-- Define the price of the book
variable (x : ℝ)

-- Define the conditions based on the wrong guesses
def student_A_wrong : Prop := ¬(x ≥ 15)
def student_B_wrong : Prop := ¬(x ≤ 12)
def student_C_wrong : Prop := ¬(x ≤ 10)

-- Theorem statement
theorem book_price_range 
  (hA : student_A_wrong x)
  (hB : student_B_wrong x)
  (hC : student_C_wrong x) :
  12 < x ∧ x < 15 := by
  sorry

end NUMINAMATH_CALUDE_book_price_range_l2032_203254


namespace NUMINAMATH_CALUDE_initial_boys_on_slide_l2032_203267

theorem initial_boys_on_slide (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  additional = 13 → total = 35 → initial + additional = total → initial = 22 := by
sorry

end NUMINAMATH_CALUDE_initial_boys_on_slide_l2032_203267


namespace NUMINAMATH_CALUDE_orange_business_profit_l2032_203287

/-- Represents the profit calculation for Mr. Smith's orange business --/
theorem orange_business_profit :
  let small_oranges : ℕ := 5
  let medium_oranges : ℕ := 3
  let large_oranges : ℕ := 3
  let small_buy_price : ℚ := 1
  let medium_buy_price : ℚ := 2
  let large_buy_price : ℚ := 3
  let small_sell_price : ℚ := 1.5
  let medium_sell_price : ℚ := 3
  let large_sell_price : ℚ := 4
  let transportation_cost : ℚ := 2
  let storage_fee : ℚ := 1
  
  let total_buy_cost : ℚ := 
    small_oranges * small_buy_price + 
    medium_oranges * medium_buy_price + 
    large_oranges * large_buy_price +
    transportation_cost + storage_fee
  
  let total_sell_revenue : ℚ :=
    small_oranges * small_sell_price +
    medium_oranges * medium_sell_price +
    large_oranges * large_sell_price
  
  let profit : ℚ := total_sell_revenue - total_buy_cost
  
  profit = 5.5 := by sorry

end NUMINAMATH_CALUDE_orange_business_profit_l2032_203287
