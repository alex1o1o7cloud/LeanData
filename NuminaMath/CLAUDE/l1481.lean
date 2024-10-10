import Mathlib

namespace decimal_to_fraction_l1481_148140

theorem decimal_to_fraction : 
  ∃ (n d : ℤ), d ≠ 0 ∧ 3.75 = (n : ℚ) / (d : ℚ) ∧ n = 15 ∧ d = 4 :=
by sorry

end decimal_to_fraction_l1481_148140


namespace cos_two_theta_value_l1481_148165

theorem cos_two_theta_value (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 4) : 
  Real.cos (2 * θ) = 1/8 := by
  sorry

end cos_two_theta_value_l1481_148165


namespace investment_ratio_l1481_148102

def investment_x : ℤ := 5000
def investment_y : ℤ := 15000

theorem investment_ratio :
  (investment_x : ℚ) / investment_y = 1 / 3 := by sorry

end investment_ratio_l1481_148102


namespace zd_length_l1481_148106

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  let xy := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let yz := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  let xz := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  xy = 8 ∧ yz = 15 ∧ xz = 17

-- Define the angle bisector ZD
def AngleBisector (X Y Z D : ℝ × ℝ) : Prop :=
  let xd := Real.sqrt ((X.1 - D.1)^2 + (X.2 - D.2)^2)
  let yd := Real.sqrt ((Y.1 - D.1)^2 + (Y.2 - D.2)^2)
  let xz := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  let yz := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  xd / yd = xz / yz

-- Theorem statement
theorem zd_length (X Y Z D : ℝ × ℝ) : 
  Triangle X Y Z → AngleBisector X Y Z D → 
  Real.sqrt ((Z.1 - D.1)^2 + (Z.2 - D.2)^2) = Real.sqrt 284.484375 :=
by sorry

end zd_length_l1481_148106


namespace special_function_inequality_l1481_148111

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  differentiable : Differentiable ℝ f
  greater_than_derivative : ∀ x, f x > deriv f x
  initial_value : f 0 = 1

/-- The main theorem -/
theorem special_function_inequality (F : SpecialFunction) :
  ∀ x, (F.f x / Real.exp x < 1) ↔ x > 0 := by
  sorry

end special_function_inequality_l1481_148111


namespace classmate_heights_most_suitable_l1481_148138

/-- Represents a survey option -/
inductive SurveyOption
  | LightBulbs
  | RiverWater
  | TVViewership
  | ClassmateHeights

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  accessibility : Bool
  non_destructive : Bool

/-- Defines what makes a survey comprehensive -/
def is_comprehensive (s : SurveyCharacteristics) : Prop :=
  s.population_size < 1000 ∧ s.accessibility ∧ s.non_destructive

/-- Assigns characteristics to each survey option -/
def survey_properties : SurveyOption → SurveyCharacteristics
  | SurveyOption.LightBulbs => ⟨100, true, false⟩
  | SurveyOption.RiverWater => ⟨10000, false, true⟩
  | SurveyOption.TVViewership => ⟨1000000, false, true⟩
  | SurveyOption.ClassmateHeights => ⟨30, true, true⟩

/-- Theorem stating that surveying classmate heights is the most suitable for a comprehensive survey -/
theorem classmate_heights_most_suitable :
  ∀ (s : SurveyOption), s ≠ SurveyOption.ClassmateHeights →
  ¬(is_comprehensive (survey_properties s)) ∧
  (is_comprehensive (survey_properties SurveyOption.ClassmateHeights)) :=
sorry


end classmate_heights_most_suitable_l1481_148138


namespace sum_of_squares_16_to_30_l1481_148170

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8215 :=
by
  sorry

end sum_of_squares_16_to_30_l1481_148170


namespace total_maggots_served_l1481_148110

def feeding_1 : ℕ := 10
def feeding_2 : ℕ := 15
def feeding_3 : ℕ := 2 * feeding_2
def feeding_4 : ℕ := feeding_3 - 5

theorem total_maggots_served :
  feeding_1 + feeding_2 + feeding_3 + feeding_4 = 80 := by
  sorry

end total_maggots_served_l1481_148110


namespace points_are_coplanar_l1481_148142

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem points_are_coplanar
  (h_not_collinear : ¬ ∃ (k : ℝ), e₂ = k • e₁)
  (h_AB : B - A = e₁ + e₂)
  (h_AC : C - A = 2 • e₁ + 8 • e₂)
  (h_AD : D - A = 3 • e₁ - 5 • e₂) :
  ∃ (x y : ℝ), D - A = x • (B - A) + y • (C - A) :=
sorry

end points_are_coplanar_l1481_148142


namespace count_maximal_arithmetic_sequences_correct_l1481_148193

/-- 
Given a positive integer n, count_maximal_arithmetic_sequences returns the number of 
maximal arithmetic sequences that can be formed from the set {1, 2, ..., n}.
A maximal arithmetic sequence is defined as an arithmetic sequence with a positive 
difference, containing at least two terms from the set, and to which no other element 
from the set can be added while maintaining the arithmetic progression.
-/
def count_maximal_arithmetic_sequences (n : ℕ) : ℕ :=
  (n^2) / 4

theorem count_maximal_arithmetic_sequences_correct (n : ℕ) :
  count_maximal_arithmetic_sequences n = ⌊(n^2 : ℚ) / 4⌋ := by
  sorry

#eval count_maximal_arithmetic_sequences 10  -- Expected output: 25

end count_maximal_arithmetic_sequences_correct_l1481_148193


namespace equation_solutions_l1481_148144

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (1/2 * (x₁ - 3)^2 = 18 ∧ x₁ = 9) ∧
                (1/2 * (x₂ - 3)^2 = 18 ∧ x₂ = -3)) ∧
  (∃ y₁ y₂ : ℝ, (y₁^2 + 6*y₁ = 5 ∧ y₁ = -3 + Real.sqrt 14) ∧
                (y₂^2 + 6*y₂ = 5 ∧ y₂ = -3 - Real.sqrt 14)) :=
by sorry

end equation_solutions_l1481_148144


namespace smallest_multiple_of_36_and_45_not_25_l1481_148131

theorem smallest_multiple_of_36_and_45_not_25 :
  ∃ (n : ℕ), n > 0 ∧ 36 ∣ n ∧ 45 ∣ n ∧ ¬(25 ∣ n) ∧
  ∀ (m : ℕ), m > 0 → 36 ∣ m → 45 ∣ m → ¬(25 ∣ m) → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_multiple_of_36_and_45_not_25_l1481_148131


namespace mets_to_red_sox_ratio_l1481_148159

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The given conditions of the problem -/
def fan_problem (fc : FanCounts) : Prop :=
  fc.yankees * 2 = fc.mets * 3 ∧  -- Ratio of Yankees to Mets is 3:2
  fc.yankees + fc.mets + fc.red_sox = 330 ∧  -- Total fans
  fc.mets = 88  -- Number of Mets fans

/-- The theorem to prove -/
theorem mets_to_red_sox_ratio 
  (fc : FanCounts) 
  (h : fan_problem fc) : 
  ∃ (r : Ratio), r.numerator = 4 ∧ r.denominator = 5 ∧
  r.numerator * fc.red_sox = r.denominator * fc.mets :=
sorry

end mets_to_red_sox_ratio_l1481_148159


namespace prob_same_club_is_one_third_l1481_148151

/-- The number of clubs -/
def num_clubs : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 2

/-- The probability of two students joining the same club given equal probability of joining any club -/
def prob_same_club : ℚ := 1 / 3

/-- Theorem stating that the probability of two students joining the same club is 1/3 -/
theorem prob_same_club_is_one_third :
  prob_same_club = 1 / 3 := by sorry

end prob_same_club_is_one_third_l1481_148151


namespace ratio_sum_in_triangle_l1481_148194

/-- Given a triangle ABC with the following properties:
  - B is the midpoint of AC
  - D divides BC such that BD:DC = 2:1
  - E divides AB such that AE:EB = 1:3
  This theorem proves that the sum of the ratios EF/FC + AF/FD equals 13/4 -/
theorem ratio_sum_in_triangle (A B C D E F : ℝ × ℝ) : 
  let midpoint (P Q : ℝ × ℝ) := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let divide_segment (P Q : ℝ × ℝ) (r s : ℝ) := 
    ((r * Q.1 + s * P.1) / (r + s), (r * Q.2 + s * P.2) / (r + s))
  B = midpoint A C ∧
  D = divide_segment B C 2 1 ∧
  E = divide_segment A B 1 3 →
  let EF := ‖E - F‖
  let FC := ‖F - C‖
  let AF := ‖A - F‖
  let FD := ‖F - D‖
  EF / FC + AF / FD = 13 / 4 := by
  sorry

end ratio_sum_in_triangle_l1481_148194


namespace circle_equation_from_line_l1481_148160

/-- Given a line in polar coordinates that intersects the polar axis, 
    find the polar equation of the circle with the intersection point's diameter --/
theorem circle_equation_from_line (θ : Real) (ρ p : Real → Real) :
  (∀ θ, p θ * Real.cos θ - 2 = 0) →  -- Line equation
  (∃ M : Real × Real, M.1 = 2 ∧ M.2 = 0) →  -- Intersection point
  (∀ θ, ρ θ = 2 * Real.cos θ) :=  -- Circle equation
by sorry

end circle_equation_from_line_l1481_148160


namespace number_less_than_l1481_148100

theorem number_less_than : (0.86 : ℝ) - 0.82 = 0.04 := by
  sorry

end number_less_than_l1481_148100


namespace f_increasing_on_interval_l1481_148163

/-- The function f(x) = x * e^(-x) is increasing on (-∞, 1) -/
theorem f_increasing_on_interval (x : ℝ) : x < 1 → Monotone (fun x => x * Real.exp (-x)) := by
  sorry

end f_increasing_on_interval_l1481_148163


namespace tan_over_cos_squared_l1481_148183

theorem tan_over_cos_squared (α : Real) (P : ℝ × ℝ) :
  P = (-1, 2) →
  (∃ r : ℝ, r > 0 ∧ P = (r * Real.cos α, r * Real.sin α)) →
  Real.tan α / (Real.cos α)^2 = -10 :=
by sorry

end tan_over_cos_squared_l1481_148183


namespace arithmetic_expression_evaluation_l1481_148119

theorem arithmetic_expression_evaluation : 7 + 15 / 3 - 5 * 2 = 2 := by
  sorry

end arithmetic_expression_evaluation_l1481_148119


namespace quadratic_decomposition_l1481_148161

theorem quadratic_decomposition :
  ∃ (k : ℤ) (a : ℝ), ∀ y : ℝ, y^2 + 14*y + 60 = (y + a)^2 + k ∧ k = 11 := by
sorry

end quadratic_decomposition_l1481_148161


namespace f_positive_iff_x_range_l1481_148101

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem f_positive_iff_x_range :
  (∀ x : ℝ, (∀ a ∈ Set.Icc (-1 : ℝ) 1, f x a > 0)) ↔
  (∀ x : ℝ, x < 1 ∨ x > 3) :=
sorry

end f_positive_iff_x_range_l1481_148101


namespace absolute_value_simplification_l1481_148124

theorem absolute_value_simplification : |(-4^2 + 6)| = 10 := by
  sorry

end absolute_value_simplification_l1481_148124


namespace fraction_equality_l1481_148105

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / ((a : ℚ) + 35) = 7 / 10 → a = 82 := by
  sorry

end fraction_equality_l1481_148105


namespace extremum_condition_l1481_148174

/-- The function f(x) = ax^3 + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The function f has an extremum -/
def has_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y ∨ f a x ≥ f a y

/-- The necessary and sufficient condition for f to have an extremum is a < 0 -/
theorem extremum_condition (a : ℝ) :
  has_extremum a ↔ a < 0 :=
sorry

end extremum_condition_l1481_148174


namespace minimum_cookies_cookies_exist_l1481_148152

theorem minimum_cookies (b : ℕ) : b ≡ 5 [ZMOD 6] ∧ b ≡ 7 [ZMOD 8] ∧ b ≡ 8 [ZMOD 9] → b ≥ 239 := by
  sorry

theorem cookies_exist : ∃ b : ℕ, b ≡ 5 [ZMOD 6] ∧ b ≡ 7 [ZMOD 8] ∧ b ≡ 8 [ZMOD 9] ∧ b = 239 := by
  sorry

end minimum_cookies_cookies_exist_l1481_148152


namespace sqrt_representation_l1481_148175

theorem sqrt_representation (n : ℕ+) :
  (∃ (x : ℝ), x > 0 ∧ x^2 = n ∧ x = Real.sqrt (Real.sqrt n)) ↔ n = 1 ∧
  (∀ (x : ℝ), x > 0 ∧ x^2 = n → ∃ (m k : ℕ+), x = (k : ℝ) ^ (1 / m : ℝ)) :=
by sorry

end sqrt_representation_l1481_148175


namespace point_in_third_quadrant_l1481_148169

theorem point_in_third_quadrant :
  let angle : ℝ := 2007 * Real.pi / 180
  (Real.cos angle < 0) ∧ (Real.sin angle < 0) :=
by
  sorry

end point_in_third_quadrant_l1481_148169


namespace quadratic_roots_relation_l1481_148181

theorem quadratic_roots_relation (p q : ℝ) : 
  (∃ r s : ℝ, (2 * r^2 - 4 * r - 5 = 0) ∧ 
               (2 * s^2 - 4 * s - 5 = 0) ∧ 
               ((r + 3)^2 + p * (r + 3) + q = 0) ∧ 
               ((s + 3)^2 + p * (s + 3) + q = 0)) →
  q = 25/2 := by
sorry

end quadratic_roots_relation_l1481_148181


namespace triangle_area_coefficient_product_l1481_148109

/-- Given a triangle in the first quadrant bounded by the coordinate axes and a line,
    prove that if the area is 9, then the product of the coefficients is 4/3. -/
theorem triangle_area_coefficient_product (a b : ℝ) : 
  a > 0 → b > 0 → (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 2*a*x + 3*b*y ≤ 12) → 
  (1/2 * (12/(2*a)) * (12/(3*b)) = 9) → a * b = 4/3 := by
sorry

end triangle_area_coefficient_product_l1481_148109


namespace intersection_of_A_and_B_l1481_148154

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| ≤ 2}
def B : Set ℝ := {x : ℝ | 3*x - 2 ≥ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l1481_148154


namespace at_least_two_positive_l1481_148185

theorem at_least_two_positive (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c > 0) (h_prod : a * b + b * c + c * a > 0) :
  (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (c > 0 ∧ a > 0) :=
sorry

end at_least_two_positive_l1481_148185


namespace purely_imaginary_fraction_l1481_148103

theorem purely_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + 2 * Complex.I) / (1 + Complex.I) = b * Complex.I) → a = -2 := by
  sorry

end purely_imaginary_fraction_l1481_148103


namespace correct_number_of_small_boxes_l1481_148120

-- Define the number of chocolate bars in each small box
def chocolates_per_small_box : ℕ := 26

-- Define the total number of chocolate bars in the large box
def total_chocolates : ℕ := 442

-- Define the number of small boxes in the large box
def num_small_boxes : ℕ := total_chocolates / chocolates_per_small_box

-- Theorem statement
theorem correct_number_of_small_boxes : num_small_boxes = 17 := by
  sorry

end correct_number_of_small_boxes_l1481_148120


namespace isosceles_triangles_count_l1481_148130

/-- A right hexagonal prism with height 2 and regular hexagonal bases of side length 1 -/
structure HexagonalPrism where
  height : ℝ
  base_side_length : ℝ
  height_eq : height = 2
  side_eq : base_side_length = 1

/-- A triangle formed by three vertices of the hexagonal prism -/
structure PrismTriangle where
  prism : HexagonalPrism
  v1 : Fin 12
  v2 : Fin 12
  v3 : Fin 12
  distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

/-- Predicate to determine if a triangle is isosceles -/
def is_isosceles (t : PrismTriangle) : Prop :=
  sorry

/-- The number of isosceles triangles in the hexagonal prism -/
def num_isosceles_triangles (p : HexagonalPrism) : ℕ :=
  sorry

/-- Theorem stating that the number of isosceles triangles is 24 -/
theorem isosceles_triangles_count (p : HexagonalPrism) :
  num_isosceles_triangles p = 24 :=
sorry

end isosceles_triangles_count_l1481_148130


namespace megans_hourly_wage_l1481_148116

/-- Megan's hourly wage problem -/
theorem megans_hourly_wage (hours_per_day : ℕ) (days_per_month : ℕ) (earnings_two_months : ℕ) 
  (h1 : hours_per_day = 8)
  (h2 : days_per_month = 20)
  (h3 : earnings_two_months = 2400) :
  (earnings_two_months : ℚ) / (2 * days_per_month * hours_per_day) = 15/2 := by
  sorry

#eval (2400 : ℚ) / (2 * 20 * 8) -- This should evaluate to 7.5

end megans_hourly_wage_l1481_148116


namespace gcd_digits_bound_l1481_148176

theorem gcd_digits_bound (a b : ℕ) : 
  1000000 ≤ a ∧ a < 10000000 ∧
  1000000 ≤ b ∧ b < 10000000 ∧
  1000000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10000000000000 →
  Nat.gcd a b < 100 := by
sorry

end gcd_digits_bound_l1481_148176


namespace intersection_of_A_and_B_l1481_148191

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 5, 6}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by sorry

end intersection_of_A_and_B_l1481_148191


namespace f_range_l1481_148189

noncomputable def f (x : ℝ) : ℝ := x^2 / (Real.log x + x)

noncomputable def g (x : ℝ) : ℝ := Real.log x + x

theorem f_range :
  ∃ (a : ℝ), 0 < a ∧ a < 1 ∧ g a = 0 →
  Set.range f = {y | y < 0 ∨ y ≥ 1} :=
sorry

end f_range_l1481_148189


namespace factorial_vs_power_l1481_148145

theorem factorial_vs_power : 100^200 > Nat.factorial 200 := by
  sorry

end factorial_vs_power_l1481_148145


namespace problem_solution_l1481_148137

theorem problem_solution (a : ℝ) (h : a = 1 / (Real.sqrt 2 - 1)) : 4 * a^2 - 8 * a - 3 = 1 := by
  sorry

end problem_solution_l1481_148137


namespace log_product_theorem_l1481_148115

-- Define the exponent rule
axiom exponent_rule {a : ℝ} (m n : ℝ) : a^m * a^n = a^(m + n)

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_product_theorem (b x y : ℝ) (hb : b > 0) (hb1 : b ≠ 1) (hx : x > 0) (hy : y > 0) :
  log b (x * y) = log b x + log b y :=
sorry

end log_product_theorem_l1481_148115


namespace julie_rowing_distance_l1481_148122

theorem julie_rowing_distance (downstream_distance : ℝ) (time : ℝ) (stream_speed : ℝ) 
  (h1 : downstream_distance = 72)
  (h2 : time = 4)
  (h3 : stream_speed = 0.5) :
  ∃ (upstream_distance : ℝ), 
    upstream_distance = 68 ∧ 
    time = upstream_distance / (downstream_distance / (2 * time) - stream_speed) ∧
    time = downstream_distance / (downstream_distance / (2 * time) + stream_speed) :=
by sorry

end julie_rowing_distance_l1481_148122


namespace geometric_sequence_sum_l1481_148157

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem about the sum of a_3 and a_5 in a specific geometric sequence. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  IsGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l1481_148157


namespace problem_solution_l1481_148196

theorem problem_solution :
  (2017^2 - 2016 * 2018 = 1) ∧
  (∀ a b : ℤ, a + b = 7 → a * b = -1 → 
    ((a + b)^2 = 49) ∧ (a^2 - 3*a*b + b^2 = 54)) := by
  sorry

end problem_solution_l1481_148196


namespace crank_slider_motion_l1481_148167

/-- Crank-slider mechanism parameters -/
structure CrankSlider where
  OA : ℝ
  AB : ℝ
  ω : ℝ
  AM : ℝ

/-- Position and velocity of point M -/
structure PointM where
  x : ℝ → ℝ
  y : ℝ → ℝ
  vx : ℝ → ℝ
  vy : ℝ → ℝ

/-- Theorem stating the equations of motion for point M -/
theorem crank_slider_motion (cs : CrankSlider) (t : ℝ) : 
  cs.OA = 90 ∧ cs.AB = 90 ∧ cs.ω = 10 ∧ cs.AM = 60 →
  ∃ (pm : PointM),
    pm.x t = 90 * Real.cos (10 * t) - 60 * Real.sin (10 * t) ∧
    pm.y t = 90 * Real.sin (10 * t) - 60 * Real.cos (10 * t) ∧
    pm.vx t = -900 * Real.sin (10 * t) - 600 * Real.cos (10 * t) ∧
    pm.vy t = 900 * Real.cos (10 * t) + 600 * Real.sin (10 * t) := by
  sorry


end crank_slider_motion_l1481_148167


namespace exists_n_satisfying_conditions_l1481_148197

/-- The number of distinct prime factors of n -/
def omega (n : ℕ) : ℕ := sorry

/-- The sum of the exponents in the prime factorization of n -/
def Omega (n : ℕ) : ℕ := sorry

/-- For any fixed positive integer k and positive reals α and β,
    there exists a positive integer n > 1 satisfying the given conditions -/
theorem exists_n_satisfying_conditions (k : ℕ) (α β : ℝ) 
    (hk : k > 0) (hα : α > 0) (hβ : β > 0) :
  ∃ n : ℕ, n > 1 ∧ 
    (omega (n + k) : ℝ) / (omega n) > α ∧
    (Omega (n + k) : ℝ) / (Omega n) < β := by
  sorry

end exists_n_satisfying_conditions_l1481_148197


namespace population_growth_proof_l1481_148172

/-- The percentage increase in population during the first year -/
def first_year_increase : ℝ := 25

/-- The initial population two years ago -/
def initial_population : ℝ := 1200

/-- The population after two years of growth -/
def final_population : ℝ := 1950

/-- The percentage increase in population during the second year -/
def second_year_increase : ℝ := 30

theorem population_growth_proof :
  initial_population * (1 + first_year_increase / 100) * (1 + second_year_increase / 100) = final_population :=
sorry

end population_growth_proof_l1481_148172


namespace caden_coin_value_l1481_148113

/-- Represents the number of coins of each type Caden has -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in dollars -/
def total_value (coins : CoinCounts) : ℚ :=
  (coins.pennies : ℚ) / 100 +
  (coins.nickels : ℚ) / 20 +
  (coins.dimes : ℚ) / 10 +
  (coins.quarters : ℚ) / 4

/-- Theorem stating that Caden's coins total $8.00 -/
theorem caden_coin_value :
  ∀ (coins : CoinCounts),
    coins.pennies = 120 →
    coins.nickels = coins.pennies / 3 →
    coins.dimes = coins.nickels / 5 →
    coins.quarters = 2 * coins.dimes →
    total_value coins = 8 := by
  sorry

end caden_coin_value_l1481_148113


namespace odd_digits_base4_157_l1481_148178

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of odd digits in the base-4 representation of 157 is 3 -/
theorem odd_digits_base4_157 : countOddDigits (toBase4 157) = 3 :=
  sorry

end odd_digits_base4_157_l1481_148178


namespace maggies_portion_l1481_148179

theorem maggies_portion (total : ℝ) (maggies_share : ℝ) (debbys_portion : ℝ) :
  total = 6000 →
  maggies_share = 4500 →
  debbys_portion = 0.25 →
  maggies_share / total = 0.75 := by
  sorry

end maggies_portion_l1481_148179


namespace matrix_cube_computation_l1481_148129

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; 2, 0]

theorem matrix_cube_computation :
  A ^ 3 = !![(-8), 0; 0, (-8)] := by sorry

end matrix_cube_computation_l1481_148129


namespace B_2_1_eq_12_l1481_148128

def B : ℕ → ℕ → ℕ
  | 0, n => n + 2
  | m + 1, 0 => B m 2
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_1_eq_12 : B 2 1 = 12 := by
  sorry

end B_2_1_eq_12_l1481_148128


namespace factorize_difference_of_squares_factorize_polynomial_l1481_148112

-- Problem 1
theorem factorize_difference_of_squares (x y : ℝ) :
  4 * x^2 - 25 * y^2 = (2*x + 5*y) * (2*x - 5*y) := by
  sorry

-- Problem 2
theorem factorize_polynomial (x y : ℝ) :
  -3 * x * y^3 + 27 * x^3 * y = -3 * x * y * (y + 3*x) * (y - 3*x) := by
  sorry

end factorize_difference_of_squares_factorize_polynomial_l1481_148112


namespace expression_equality_l1481_148186

theorem expression_equality : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end expression_equality_l1481_148186


namespace inequality_solution_set_l1481_148117

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 ∨ x = 6 :=
by sorry

end inequality_solution_set_l1481_148117


namespace turtleneck_discount_l1481_148173

theorem turtleneck_discount (C : ℝ) (C_pos : C > 0) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_profit := 0.41
  let initial_price := C * (1 + initial_markup)
  let new_year_price := initial_price * (1 + new_year_markup)
  let february_price := C * (1 + february_profit)
  let discount := 1 - (february_price / new_year_price)
  discount = 0.06 := by
sorry

end turtleneck_discount_l1481_148173


namespace rectangle_longer_side_l1481_148190

/-- Given a circle with radius 5 cm tangent to three sides of a rectangle, 
    and the area of the rectangle being three times the area of the circle,
    prove that the length of the longer side of the rectangle is 7.5π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) 
  (h1 : circle_radius = 5)
  (h2 : rectangle_area = 3 * π * circle_radius^2) : 
  rectangle_area / (2 * circle_radius) = 7.5 * π := by
  sorry

end rectangle_longer_side_l1481_148190


namespace line_slope_intercept_sum_line_slope_intercept_sum_proof_l1481_148199

/-- Given two points A(1, 4) and B(5, 16) on a line, 
    the sum of the line's slope and y-intercept is 4. -/
theorem line_slope_intercept_sum : ℝ → ℝ → Prop :=
  fun (slope : ℝ) (y_intercept : ℝ) =>
    (slope * 1 + y_intercept = 4) ∧  -- Point A satisfies the line equation
    (slope * 5 + y_intercept = 16) ∧ -- Point B satisfies the line equation
    (slope + y_intercept = 4)        -- Sum of slope and y-intercept is 4

/-- Proof of the theorem -/
theorem line_slope_intercept_sum_proof : ∃ (slope : ℝ) (y_intercept : ℝ), 
  line_slope_intercept_sum slope y_intercept := by
  sorry

end line_slope_intercept_sum_line_slope_intercept_sum_proof_l1481_148199


namespace sine_cosine_sum_simplification_l1481_148141

theorem sine_cosine_sum_simplification (x y : ℝ) : 
  Real.sin (x - 2*y) * Real.cos (3*y) + Real.cos (x - 2*y) * Real.sin (3*y) = Real.sin (x + y) := by
  sorry

end sine_cosine_sum_simplification_l1481_148141


namespace gcd_lcm_problem_l1481_148123

theorem gcd_lcm_problem (a b : ℕ+) : 
  Nat.gcd a b = 21 ∧ Nat.lcm a b = 3969 → 
  (a = 21 ∧ b = 3969) ∨ (a = 147 ∧ b = 567) ∨ (a = 3969 ∧ b = 21) ∨ (a = 567 ∧ b = 147) :=
by sorry

end gcd_lcm_problem_l1481_148123


namespace meaningful_square_root_range_l1481_148198

theorem meaningful_square_root_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (2 - x)) ↔ x < 2 := by
sorry

end meaningful_square_root_range_l1481_148198


namespace average_bracelets_per_day_l1481_148192

def bike_cost : ℕ := 112
def selling_weeks : ℕ := 2
def bracelet_price : ℕ := 1
def days_per_week : ℕ := 7

theorem average_bracelets_per_day :
  (bike_cost / (selling_weeks * days_per_week)) / bracelet_price = 8 :=
by sorry

end average_bracelets_per_day_l1481_148192


namespace equal_probability_red_black_l1481_148162

/-- Represents a deck of cards after removing face cards and 8's --/
structure Deck :=
  (total_cards : ℕ)
  (red_divisible_by_3 : ℕ)
  (black_divisible_by_3 : ℕ)

/-- Represents the probability of picking a card of a certain color divisible by 3 --/
def probability_divisible_by_3 (deck : Deck) (color : String) : ℚ :=
  if color = "red" then
    (deck.red_divisible_by_3 : ℚ) / deck.total_cards
  else if color = "black" then
    (deck.black_divisible_by_3 : ℚ) / deck.total_cards
  else
    0

/-- The main theorem stating that the probabilities are equal for red and black cards --/
theorem equal_probability_red_black (deck : Deck) 
    (h1 : deck.total_cards = 36)
    (h2 : deck.red_divisible_by_3 = 6)
    (h3 : deck.black_divisible_by_3 = 6) :
  probability_divisible_by_3 deck "red" = probability_divisible_by_3 deck "black" :=
by
  sorry

#check equal_probability_red_black

end equal_probability_red_black_l1481_148162


namespace seed_germination_rate_l1481_148108

theorem seed_germination_rate (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot2 overall_germination_rate : ℝ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot2 = 35 →
  overall_germination_rate = 28.999999999999996 →
  (((overall_germination_rate / 100) * (seeds_plot1 + seeds_plot2) - 
    (germination_rate_plot2 / 100) * seeds_plot2) / seeds_plot1) * 100 = 25 := by
  sorry

end seed_germination_rate_l1481_148108


namespace sin_80_in_terms_of_tan_100_l1481_148184

theorem sin_80_in_terms_of_tan_100 (k : ℝ) (h : Real.tan (100 * π / 180) = k) :
  Real.sin (80 * π / 180) = -k / Real.sqrt (1 + k^2) := by
  sorry

end sin_80_in_terms_of_tan_100_l1481_148184


namespace range_of_m_l1481_148114

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x^4 - x^2 + 1) / x^2 > m

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (-(5-2*m))^y < (-(5-2*m))^x

-- Define the theorem
theorem range_of_m :
  (∀ m : ℝ, (p m ∨ q m)) ∧ (¬∀ m : ℝ, (p m ∧ q m)) →
  ∃ a b : ℝ, a = 1 ∧ b = 2 ∧ ∀ m : ℝ, a ≤ m ∧ m < b :=
sorry

end range_of_m_l1481_148114


namespace guard_circles_l1481_148134

/-- Calculates the number of times a guard should circle a rectangular warehouse --/
def warehouseCircles (length width walked skipped : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let actualCircles := walked / perimeter
  actualCircles + skipped

/-- Theorem stating that for the given warehouse and guard's walk, the number of circles is 10 --/
theorem guard_circles : 
  warehouseCircles 600 400 16000 2 = 10 := by sorry

end guard_circles_l1481_148134


namespace first_group_size_l1481_148155

/-- The number of persons in the first group that can repair a road -/
def first_group : ℕ :=
  let days : ℕ := 12
  let hours_per_day_first : ℕ := 5
  let second_group : ℕ := 30
  let hours_per_day_second : ℕ := 6
  (second_group * hours_per_day_second) / hours_per_day_first

theorem first_group_size :
  first_group = 36 :=
by sorry

end first_group_size_l1481_148155


namespace quadrilateral_angle_difference_l1481_148147

/-- A quadrilateral with angles in ratio 3:4:5:6 has a difference of 60° between its largest and smallest angles -/
theorem quadrilateral_angle_difference (a b c d : ℝ) : 
  a + b + c + d = 360 →  -- Sum of angles in a quadrilateral
  ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k ∧ d = 6*k →  -- Angles in ratio 3:4:5:6
  (6*k) - (3*k) = 60 :=  -- Difference between largest and smallest angles
by sorry

end quadrilateral_angle_difference_l1481_148147


namespace successful_meeting_probability_l1481_148118

-- Define the arrival times as real numbers between 0 and 2 (representing hours after 3:00 p.m.)
variable (x y z : ℝ)

-- Define the conditions for a successful meeting
def successful_meeting (x y z : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 ∧
  0 ≤ y ∧ y ≤ 2 ∧
  0 ≤ z ∧ z ≤ 2 ∧
  z > x ∧ z > y ∧
  |x - y| ≤ 1.5

-- Define the probability space
def total_outcomes : ℝ := 8

-- Define the volume of the region where the meeting is successful
noncomputable def successful_volume : ℝ := 8/9

-- Theorem stating the probability of a successful meeting
theorem successful_meeting_probability :
  (successful_volume / total_outcomes) = 1/9 :=
sorry

end successful_meeting_probability_l1481_148118


namespace two_cars_intersection_problem_l1481_148107

/-- Two cars approaching an intersection problem -/
theorem two_cars_intersection_problem 
  (s₁ : ℝ) (s₂ : ℝ) (v₁ : ℝ) (s : ℝ)
  (h₁ : s₁ = 1600) -- Initial distance of first car
  (h₂ : s₂ = 800)  -- Initial distance of second car
  (h₃ : v₁ = 72)   -- Speed of first car in km/h
  (h₄ : s = 200)   -- Distance between cars when first car reaches intersection
  : ∃ v₂ : ℝ, (v₂ = 7.5 ∨ v₂ = 12.5) ∧ 
    v₂ * (s₁ / (v₁ * 1000 / 3600)) = s₂ - s ∨ 
    v₂ * (s₁ / (v₁ * 1000 / 3600)) = s₂ + s :=
by sorry

end two_cars_intersection_problem_l1481_148107


namespace linear_decreasing_iff_k_lt_neg_half_l1481_148126

/-- A function f: ℝ → ℝ is decreasing if for all x₁ < x₂, f(x₁) > f(x₂) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

/-- The linear function y = (2k+1)x + b -/
def f (k b : ℝ) (x : ℝ) : ℝ := (2*k + 1)*x + b

theorem linear_decreasing_iff_k_lt_neg_half (k b : ℝ) :
  IsDecreasing (f k b) ↔ k < -1/2 := by
  sorry

end linear_decreasing_iff_k_lt_neg_half_l1481_148126


namespace unique_solution_iff_k_eq_six_l1481_148188

/-- The equation (x+5)(x+2) = k + 3x has exactly one real solution if and only if k = 6 -/
theorem unique_solution_iff_k_eq_six (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
  sorry

end unique_solution_iff_k_eq_six_l1481_148188


namespace boat_downstream_distance_l1481_148136

/-- Proves the distance traveled downstream by a boat -/
theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (travel_time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : travel_time = 5) : 
  boat_speed + stream_speed * travel_time = 140 := by
  sorry

end boat_downstream_distance_l1481_148136


namespace smallest_k_with_remainders_l1481_148180

theorem smallest_k_with_remainders : ∃! k : ℕ, 
  k > 1 ∧ 
  k % 19 = 1 ∧ 
  k % 7 = 1 ∧ 
  k % 3 = 1 ∧
  ∀ m : ℕ, (m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1) → k ≤ m :=
by
  use 400
  sorry

end smallest_k_with_remainders_l1481_148180


namespace unique_solution_l1481_148153

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := lg (x + 1) = (1 / 2) * (Real.log x / Real.log 3)

-- Theorem statement
theorem unique_solution : ∃! x : ℝ, x > 0 ∧ equation x ∧ x = 9 :=
  sorry

end unique_solution_l1481_148153


namespace inequality_solution_set_l1481_148143

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 + 2*x < 3

-- Define the solution set
def solution_set : Set ℝ := {x | -3 < x ∧ x < 1}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry

end inequality_solution_set_l1481_148143


namespace triangle_side_difference_is_12_l1481_148168

def triangle_side_difference (y : ℤ) : Prop :=
  ∃ (a b : ℤ), 
    a = 7 ∧ b = 9 ∧  -- Given side lengths
    y > |a - b| ∧    -- Triangle inequality lower bound
    y < a + b ∧      -- Triangle inequality upper bound
    y ≥ 3 ∧ y ≤ 15   -- Integral bounds for y

theorem triangle_side_difference_is_12 : 
  (∀ y : ℤ, triangle_side_difference y → y ≤ 15) ∧ 
  (∀ y : ℤ, triangle_side_difference y → y ≥ 3) ∧
  (15 - 3 = 12) :=
sorry

end triangle_side_difference_is_12_l1481_148168


namespace complex_equality_modulus_l1481_148182

theorem complex_equality_modulus (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (1 + a * i) * i = 2 - b * i →
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end complex_equality_modulus_l1481_148182


namespace parallel_to_a_l1481_148177

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

/-- The vector a is defined as (-5, 4) -/
def a : ℝ × ℝ := (-5, 4)

/-- Theorem: A vector (x, y) is parallel to a = (-5, 4) if and only if
    there exists a real number k such that (x, y) = (-5k, 4k) -/
theorem parallel_to_a (x y : ℝ) :
  parallel (x, y) a ↔ ∃ k : ℝ, (x, y) = (-5 * k, 4 * k) :=
sorry

end parallel_to_a_l1481_148177


namespace diamond_three_four_l1481_148149

def diamond (a b : ℝ) : ℝ := a^2 * b^2 - b + 2

theorem diamond_three_four : diamond 3 4 = 142 := by
  sorry

end diamond_three_four_l1481_148149


namespace b_is_composite_greatest_number_of_factors_l1481_148125

/-- The greatest number of positive factors for b^m -/
def max_factors : ℕ := 81

/-- b is a positive integer less than or equal to 20 -/
def b : ℕ := 16

/-- m is a positive integer less than or equal to 20 -/
def m : ℕ := 20

/-- b is composite -/
theorem b_is_composite : ¬ Nat.Prime b := by sorry

theorem greatest_number_of_factors :
  ∀ b' m' : ℕ,
  b' ≤ 20 → m' ≤ 20 → b' > 1 → ¬ Nat.Prime b' →
  (Nat.divisors (b' ^ m')).card ≤ max_factors := by sorry

end b_is_composite_greatest_number_of_factors_l1481_148125


namespace smallest_unachievable_score_l1481_148132

def dart_scores : Set ℕ := {0, 1, 3, 8, 12}

def is_achievable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ dart_scores ∧ b ∈ dart_scores ∧ c ∈ dart_scores ∧ a + b + c = n

theorem smallest_unachievable_score :
  (∀ m < 22, is_achievable m) ∧ ¬is_achievable 22 :=
sorry

end smallest_unachievable_score_l1481_148132


namespace harrys_father_age_difference_l1481_148146

/-- Proves that Harry's father is 24 years older than Harry given the problem conditions -/
theorem harrys_father_age_difference : 
  ∀ (harry_age father_age mother_age : ℕ),
    harry_age = 50 →
    father_age > harry_age →
    mother_age = harry_age + 22 →
    father_age = mother_age + harry_age / 25 →
    father_age - harry_age = 24 :=
by
  sorry

end harrys_father_age_difference_l1481_148146


namespace smallest_positive_solution_l1481_148164

theorem smallest_positive_solution (x : ℝ) :
  (x > 0 ∧ x / 7 + 2 / (7 * x) = 1) → x = (7 - Real.sqrt 41) / 2 := by
  sorry

end smallest_positive_solution_l1481_148164


namespace complex_magnitude_product_l1481_148166

theorem complex_magnitude_product : Complex.abs (5 - 3 * Complex.I) * Complex.abs (5 + 3 * Complex.I) = 34 := by
  sorry

end complex_magnitude_product_l1481_148166


namespace copy_pages_theorem_l1481_148187

/-- Given a cost per page in cents and a budget in dollars, 
    calculate the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (budget_dollars : ℕ) : ℕ :=
  (budget_dollars * 100) / cost_per_page

/-- Theorem: With a cost of 3 cents per page and a budget of $15, 
    the maximum number of pages that can be copied is 500. -/
theorem copy_pages_theorem : max_pages_copied 3 15 = 500 := by
  sorry

end copy_pages_theorem_l1481_148187


namespace franks_allowance_l1481_148133

/-- The amount Frank had saved up -/
def savings : ℕ := 3

/-- The number of toys Frank could buy -/
def num_toys : ℕ := 5

/-- The price of each toy -/
def toy_price : ℕ := 8

/-- The amount Frank received for his allowance -/
def allowance : ℕ := 37

theorem franks_allowance :
  savings + allowance = num_toys * toy_price :=
by sorry

end franks_allowance_l1481_148133


namespace ponce_lighter_than_jalen_l1481_148150

/-- Represents the weights of three people and their relationships. -/
structure WeightProblem where
  ishmael : ℝ
  ponce : ℝ
  jalen : ℝ
  ishmael_heavier : ishmael = ponce + 20
  jalen_weight : jalen = 160
  average_weight : (ishmael + ponce + jalen) / 3 = 160

/-- Theorem stating that Ponce is 10 pounds lighter than Jalen. -/
theorem ponce_lighter_than_jalen (w : WeightProblem) : w.jalen - w.ponce = 10 := by
  sorry

#check ponce_lighter_than_jalen

end ponce_lighter_than_jalen_l1481_148150


namespace integer_roots_of_cubic_l1481_148121

def cubic_equation (x : Int) : Int :=
  x^3 - 4*x^2 - 11*x + 24

def is_root (x : Int) : Prop :=
  cubic_equation x = 0

theorem integer_roots_of_cubic :
  ∀ x : Int, is_root x ↔ x = -4 ∨ x = 3 ∨ x = 8 := by sorry

end integer_roots_of_cubic_l1481_148121


namespace intersection_of_M_and_N_l1481_148139

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end intersection_of_M_and_N_l1481_148139


namespace system_solution_l1481_148104

theorem system_solution (x y z : ℝ) : 
  x * y = 8 - x - 4 * y →
  y * z = 12 - 3 * y - 6 * z →
  x * z = 40 - 5 * x - 2 * z →
  x > 0 →
  x = 6 := by
sorry

end system_solution_l1481_148104


namespace sum_of_four_numbers_l1481_148195

theorem sum_of_four_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end sum_of_four_numbers_l1481_148195


namespace always_two_distinct_roots_find_m_value_l1481_148158

/-- The quadratic equation x^2 - (2m + 1)x - 2 = 0 -/
def quadratic (m : ℝ) (x : ℝ) : Prop :=
  x^2 - (2*m + 1)*x - 2 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (2*m + 1)^2 + 8

theorem always_two_distinct_roots (m : ℝ) :
  discriminant m > 0 :=
sorry

theorem find_m_value (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic m x₁) 
  (h₂ : quadratic m x₂) 
  (h₃ : x₁ + x₂ + x₁*x₂ = 1) :
  m = 1 :=
sorry

end always_two_distinct_roots_find_m_value_l1481_148158


namespace triangle_area_l1481_148127

/-- A triangle with side lengths 6, 8, and 10 has an area of 24 square units. -/
theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (1/2) * a * b = 24 := by
  sorry

end triangle_area_l1481_148127


namespace range_of_a_l1481_148171

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- State the theorem
theorem range_of_a :
  (∀ x ∈ Set.Ioo 0 2, f a x ≥ 0) ↔ a ∈ Set.Iic 1 :=
sorry

end range_of_a_l1481_148171


namespace parentheses_number_l1481_148148

theorem parentheses_number (x : ℤ) (h : x - (-2) = 3) : x = 1 := by
  sorry

end parentheses_number_l1481_148148


namespace line_point_sum_l1481_148156

/-- The line equation y = -1/2x + 8 --/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

/-- Point P is where the line crosses the x-axis --/
def P : ℝ × ℝ := (16, 0)

/-- Point Q is where the line crosses the y-axis --/
def Q : ℝ × ℝ := (0, 8)

/-- Point T is on line segment PQ --/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- Area of triangle POQ is twice the area of triangle TOP --/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) / 2 =
  2 * abs ((P.1 - 0) * (s - 0) - (r - 0) * (P.2 - 0)) / 2

theorem line_point_sum :
  ∀ r s : ℝ,
  line_equation r s →
  T_on_PQ r s →
  area_condition r s →
  r + s = 12 := by sorry

end line_point_sum_l1481_148156


namespace sum_of_roots_cubic_l1481_148135

theorem sum_of_roots_cubic (x : ℝ) : 
  (∃ s : ℝ, (∀ x, x^3 - x^2 - 13*x + 13 = 0 → (∃ y z : ℝ, y ≠ x ∧ z ≠ x ∧ z ≠ y ∧ 
    x + y + z = s))) → 
  (∃ s : ℝ, (∀ x, x^3 - x^2 - 13*x + 13 = 0 → (∃ y z : ℝ, y ≠ x ∧ z ≠ x ∧ z ≠ y ∧ 
    x + y + z = s)) ∧ s = 1) :=
by sorry

end sum_of_roots_cubic_l1481_148135
