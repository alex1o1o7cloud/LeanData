import Mathlib

namespace range_of_a_is_closed_interval_two_three_l2100_210082

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x - 2

def g (a x : ℝ) : ℝ := x^2 - a*x - a + 3

theorem range_of_a_is_closed_interval_two_three :
  ∃ (a : ℝ), ∀ x₁ x₂ : ℝ,
    f x₁ = 0 ∧ g a x₂ = 0 ∧ |x₁ - x₂| ≤ 1 →
    a ∈ Set.Icc 2 3 :=
by sorry

end range_of_a_is_closed_interval_two_three_l2100_210082


namespace distance_AB_is_420_main_theorem_l2100_210003

/-- Represents a person with a speed --/
structure Person where
  speed : ℝ

/-- Represents the problem setup --/
structure ProblemSetup where
  distance_AB : ℝ
  person_A : Person
  person_B : Person
  meeting_point : ℝ
  B_remaining_distance : ℝ

/-- The theorem statement --/
theorem distance_AB_is_420 (setup : ProblemSetup) : setup.distance_AB = 420 :=
  by
  have h1 : setup.person_A.speed > setup.person_B.speed := sorry
  have h2 : setup.meeting_point = setup.distance_AB - 240 := sorry
  have h3 : setup.B_remaining_distance = 120 := sorry
  have h4 : 2 * setup.person_A.speed > 2 * setup.person_B.speed := sorry
  sorry

/-- The main theorem --/
theorem main_theorem : ∃ (setup : ProblemSetup), setup.distance_AB = 420 :=
  by sorry

end distance_AB_is_420_main_theorem_l2100_210003


namespace quadratic_vertex_l2100_210040

/-- The quadratic function f(x) = -3x^2 - 2 has its vertex at (0, -2). -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x => -3 * x^2 - 2
  (∀ x, f x ≤ f 0) ∧ f 0 = -2 :=
by sorry

end quadratic_vertex_l2100_210040


namespace simplify_fraction_l2100_210071

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (75 * b^3) = 2/5 := by
  sorry

end simplify_fraction_l2100_210071


namespace seryozha_sandwich_candy_impossibility_l2100_210055

theorem seryozha_sandwich_candy_impossibility :
  ¬ ∃ (x y z : ℕ), x + 2*y + 3*z = 100 ∧ 3*x + 4*y + 5*z = 166 :=
by sorry

end seryozha_sandwich_candy_impossibility_l2100_210055


namespace cubic_polynomial_special_roots_l2100_210028

/-- A polynomial of degree 3 with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a cubic polynomial -/
structure CubicRoots where
  r : ℝ
  s : ℝ
  t : ℝ

/-- Proposition: For a cubic polynomial x^3 - 5x^2 + 2bx - c with real and positive roots,
    where one root is twice another and four times the third, c = 1000/343 -/
theorem cubic_polynomial_special_roots (p : CubicPolynomial) (roots : CubicRoots) :
  p.a = 1 ∧ p.b = -5 ∧  -- Coefficients of the polynomial
  (∀ x, x^3 - 5*x^2 + 2*p.b*x - p.c = 0 ↔ x = roots.r ∨ x = roots.s ∨ x = roots.t) ∧  -- Roots definition
  roots.r > 0 ∧ roots.s > 0 ∧ roots.t > 0 ∧  -- Roots are positive
  roots.s = 2 * roots.t ∧ roots.r = 4 * roots.t  -- Root relationships
  →
  p.c = 1000 / 343 := by
sorry

end cubic_polynomial_special_roots_l2100_210028


namespace not_one_zero_pronounced_l2100_210050

def number_of_pronounced_zeros (n : Nat) : Nat :=
  sorry -- Implementation of counting pronounced zeros

theorem not_one_zero_pronounced (n : Nat) (h : n = 83721000) : 
  number_of_pronounced_zeros n ≠ 1 := by
  sorry

end not_one_zero_pronounced_l2100_210050


namespace weitzenboeck_inequality_tetrahedron_l2100_210051

/-- A tetrahedron with edge lengths a, b, c, d, e, f and surface area S. -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  S : ℝ

/-- The Weitzenböck inequality for tetrahedra. -/
theorem weitzenboeck_inequality_tetrahedron (t : Tetrahedron) :
  t.S ≤ (Real.sqrt 3 / 6) * (t.a^2 + t.b^2 + t.c^2 + t.d^2 + t.e^2 + t.f^2) := by
  sorry

end weitzenboeck_inequality_tetrahedron_l2100_210051


namespace cement_mixture_weight_l2100_210058

theorem cement_mixture_weight :
  ∀ (W : ℚ),
  (2 / 7 : ℚ) * W +  -- Sand
  (3 / 7 : ℚ) * W +  -- Water
  (1 / 14 : ℚ) * W + -- Gravel
  (1 / 14 : ℚ) * W + -- Cement
  12 = W             -- Crushed stones
  →
  W = 84 :=
by sorry

end cement_mixture_weight_l2100_210058


namespace jim_siblings_l2100_210077

-- Define the characteristics
inductive EyeColor
| Blue
| Brown

inductive HairColor
| Blond
| Black

inductive GlassesWorn
| Yes
| No

-- Define a student structure
structure Student where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor
  glassesWorn : GlassesWorn

-- Define the list of students
def students : List Student := [
  ⟨"Benjamin", EyeColor.Blue, HairColor.Blond, GlassesWorn.Yes⟩,
  ⟨"Jim", EyeColor.Brown, HairColor.Blond, GlassesWorn.No⟩,
  ⟨"Nadeen", EyeColor.Brown, HairColor.Black, GlassesWorn.Yes⟩,
  ⟨"Austin", EyeColor.Blue, HairColor.Black, GlassesWorn.No⟩,
  ⟨"Tevyn", EyeColor.Blue, HairColor.Blond, GlassesWorn.Yes⟩,
  ⟨"Sue", EyeColor.Brown, HairColor.Blond, GlassesWorn.No⟩
]

-- Define a function to check if two students share at least one characteristic
def shareCharacteristic (s1 s2 : Student) : Prop :=
  s1.eyeColor = s2.eyeColor ∨ s1.hairColor = s2.hairColor ∨ s1.glassesWorn = s2.glassesWorn

-- Define a function to check if three students are siblings
def areSiblings (s1 s2 s3 : Student) : Prop :=
  shareCharacteristic s1 s2 ∧ shareCharacteristic s2 s3 ∧ shareCharacteristic s1 s3

-- Theorem statement
theorem jim_siblings :
  ∃ (jim sue benjamin : Student),
    jim ∈ students ∧ sue ∈ students ∧ benjamin ∈ students ∧
    jim.name = "Jim" ∧ sue.name = "Sue" ∧ benjamin.name = "Benjamin" ∧
    areSiblings jim sue benjamin ∧
    (∀ (other : Student), other ∈ students → other.name ≠ "Jim" → other.name ≠ "Sue" → other.name ≠ "Benjamin" →
      ¬(areSiblings jim sue other ∨ areSiblings jim benjamin other ∨ areSiblings sue benjamin other)) :=
sorry

end jim_siblings_l2100_210077


namespace total_carrots_eq_101_l2100_210011

/-- The number of carrots grown by Joan -/
def joan_carrots : ℕ := 29

/-- The number of watermelons grown by Joan -/
def joan_watermelons : ℕ := 14

/-- The number of carrots grown by Jessica -/
def jessica_carrots : ℕ := 11

/-- The number of cantaloupes grown by Jessica -/
def jessica_cantaloupes : ℕ := 9

/-- The number of carrots grown by Michael -/
def michael_carrots : ℕ := 37

/-- The number of carrots grown by Taylor -/
def taylor_carrots : ℕ := 24

/-- The number of cantaloupes grown by Taylor -/
def taylor_cantaloupes : ℕ := 3

/-- The total number of carrots grown by all -/
def total_carrots : ℕ := joan_carrots + jessica_carrots + michael_carrots + taylor_carrots

theorem total_carrots_eq_101 : total_carrots = 101 :=
by sorry

end total_carrots_eq_101_l2100_210011


namespace village_walk_speeds_l2100_210024

/-- Proves that given the conditions of the problem, the speeds of the two people are 2 km/h and 5 km/h respectively. -/
theorem village_walk_speeds (distance : ℝ) (speed_diff : ℝ) (time_diff : ℝ)
  (h1 : distance = 10)
  (h2 : speed_diff = 3)
  (h3 : time_diff = 3)
  (h4 : ∀ x : ℝ, distance / x = distance / (x + speed_diff) + time_diff → x = 2) :
  ∃ (speed1 speed2 : ℝ), speed1 = 2 ∧ speed2 = 5 :=
by sorry

end village_walk_speeds_l2100_210024


namespace clock_sale_second_price_l2100_210022

/-- Represents the sale and resale of a clock in a shop. -/
def ClockSale (original_cost : ℝ) : Prop :=
  let first_sale_price := 1.2 * original_cost
  let buy_back_price := 0.6 * original_cost
  let second_sale_price := 1.08 * original_cost
  (original_cost - buy_back_price = 100) ∧
  (second_sale_price = 270)

/-- Proves that the shop's second selling price of the clock is $270 given the conditions. -/
theorem clock_sale_second_price :
  ∃ (original_cost : ℝ), ClockSale original_cost :=
sorry

end clock_sale_second_price_l2100_210022


namespace tims_golf_balls_l2100_210096

-- Define the number of dozens Tim has
def tims_dozens : ℕ := 13

-- Define the number of items in a dozen
def items_per_dozen : ℕ := 12

-- Theorem to prove
theorem tims_golf_balls : tims_dozens * items_per_dozen = 156 := by
  sorry

end tims_golf_balls_l2100_210096


namespace log_equation_sum_l2100_210090

theorem log_equation_sum (A B C : ℕ+) : 
  (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) →
  (A : ℝ) * (Real.log 5 / Real.log 200) + (B : ℝ) * (Real.log 2 / Real.log 200) = C →
  A + B + C = 6 := by
sorry

end log_equation_sum_l2100_210090


namespace lock_problem_l2100_210064

/-- The number of buttons on the lock -/
def num_buttons : ℕ := 10

/-- The number of buttons that need to be pressed simultaneously -/
def buttons_to_press : ℕ := 3

/-- The time taken for each attempt in seconds -/
def time_per_attempt : ℕ := 2

/-- The total number of possible combinations -/
def total_combinations : ℕ := (num_buttons.choose buttons_to_press)

/-- The maximum time needed to try all combinations in seconds -/
def max_time : ℕ := total_combinations * time_per_attempt

/-- The average number of attempts needed -/
def avg_attempts : ℚ := (1 + total_combinations) / 2

/-- The average time needed to open the door in seconds -/
def avg_time : ℚ := avg_attempts * time_per_attempt

/-- The maximum number of attempts possible in 60 seconds -/
def max_attempts_in_minute : ℕ := 60 / time_per_attempt

/-- The probability of opening the door in less than 60 seconds -/
def prob_less_than_minute : ℚ := (max_attempts_in_minute - 1) / total_combinations

theorem lock_problem :
  (max_time = 240) ∧
  (avg_time = 121) ∧
  (prob_less_than_minute = 29 / 120) := by
  sorry

end lock_problem_l2100_210064


namespace union_eq_P_l2100_210045

def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x > 1 ∨ x < -1}

theorem union_eq_P : M ∪ P = P := by
  sorry

end union_eq_P_l2100_210045


namespace f_properties_l2100_210079

-- Define the function f(x) = x³ - 3x² - 9x
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- Define the domain
def domain : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- Theorem statement
theorem f_properties :
  (∃ (x : ℝ), x ∈ domain ∧ f x = 5 ∧ ∀ (y : ℝ), y ∈ domain → f y ≤ f x) ∧
  (∀ (m : ℝ), ∃ (x : ℝ), x ∈ domain ∧ f x < m) := by
  sorry

end f_properties_l2100_210079


namespace inequality_proof_l2100_210098

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end inequality_proof_l2100_210098


namespace largest_number_in_sample_l2100_210095

/-- Represents a systematic sampling process -/
structure SystematicSample where
  population_size : ℕ
  start : ℕ
  interval : ℕ

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (s : SystematicSample) : ℕ :=
  s.start + s.interval * ((s.population_size - s.start) / s.interval)

/-- Theorem: The largest number in the given systematic sample is 1468 -/
theorem largest_number_in_sample :
  let s : SystematicSample := ⟨1500, 18, 50⟩
  largest_sample_number s = 1468 := by
  sorry

end largest_number_in_sample_l2100_210095


namespace constant_term_of_x_minus_inverse_x_power_8_l2100_210049

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the constant term of (x - 1/x)^n
def constantTerm (n : ℕ) : ℤ :=
  if n % 2 = 0
  then (-1)^(n/2) * binomial n (n/2)
  else 0

-- Theorem statement
theorem constant_term_of_x_minus_inverse_x_power_8 :
  constantTerm 8 = 70 := by sorry

end constant_term_of_x_minus_inverse_x_power_8_l2100_210049


namespace circle_symmetry_l2100_210019

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define circle C1
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define circle C2
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- Define symmetry relation between two points with respect to the line
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  line_of_symmetry ((x1 + x2) / 2) ((y1 + y2) / 2)

-- Theorem statement
theorem circle_symmetry :
  ∀ x y : ℝ,
  (∃ x1 y1 : ℝ, circle_C1 x1 y1 ∧ symmetric_points x1 y1 x y) →
  circle_C2 x y :=
sorry

end circle_symmetry_l2100_210019


namespace partial_fraction_decomposition_l2100_210046

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 6 → x ≠ -3 →
  (4 * x - 3) / (x^2 - 3 * x - 18) = (7 / 3) / (x - 6) + (5 / 3) / (x + 3) := by
sorry

end partial_fraction_decomposition_l2100_210046


namespace red_stripes_on_fifty_flags_l2100_210091

/-- Calculates the total number of red stripes on multiple flags -/
def total_red_stripes (stripes_per_flag : ℕ) (num_flags : ℕ) : ℕ :=
  let remaining_stripes := stripes_per_flag - 1
  let red_remaining := remaining_stripes / 2
  let red_per_flag := red_remaining + 1
  red_per_flag * num_flags

/-- Theorem stating the total number of red stripes on 50 flags -/
theorem red_stripes_on_fifty_flags :
  total_red_stripes 25 50 = 650 := by
  sorry

end red_stripes_on_fifty_flags_l2100_210091


namespace circle_area_difference_radius_l2100_210033

theorem circle_area_difference_radius 
  (r₁ : ℝ) (r₂ : ℝ) (r₃ : ℝ) 
  (h₁ : r₁ = 21) (h₂ : r₂ = 31) 
  (h₃ : π * r₃^2 = π * r₂^2 - π * r₁^2) : 
  r₃ = 2 * Real.sqrt 130 := by
sorry

end circle_area_difference_radius_l2100_210033


namespace unique_solution_l2100_210027

def equation1 (x y z : ℝ) : Prop := x^2 - 22*y - 69*z + 703 = 0
def equation2 (x y z : ℝ) : Prop := y^2 + 23*x + 23*z - 1473 = 0
def equation3 (x y z : ℝ) : Prop := z^2 - 63*x + 66*y + 2183 = 0

theorem unique_solution :
  ∃! (x y z : ℝ), equation1 x y z ∧ equation2 x y z ∧ equation3 x y z ∧ x = 20 ∧ y = -22 ∧ z = 23 := by
  sorry

end unique_solution_l2100_210027


namespace trajectory_is_parabola_line_passes_through_fixed_point_l2100_210052

/-- A circle that passes through (1, 0) and is tangent to x = -1 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_F : (center.1 - 1)^2 + center.2^2 = radius^2
  tangent_to_l : center.1 + radius = 1

/-- The trajectory of the center of the TangentCircle -/
def trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Two distinct points on the trajectory, neither being the origin -/
structure TrajectoryPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  A_on_trajectory : A ∈ trajectory
  B_on_trajectory : B ∈ trajectory
  A_not_origin : A ≠ (0, 0)
  B_not_origin : B ≠ (0, 0)
  A_ne_B : A ≠ B
  y_product_ne_neg16 : A.2 * B.2 ≠ -16

theorem trajectory_is_parabola (c : TangentCircle) : c.center ∈ trajectory := by sorry

theorem line_passes_through_fixed_point (p : TrajectoryPoints) :
  ∃ t : ℝ, t * (p.B.1 - p.A.1) + p.A.1 = 4 ∧ t * (p.B.2 - p.A.2) + p.A.2 = 0 := by sorry

end trajectory_is_parabola_line_passes_through_fixed_point_l2100_210052


namespace quadratic_complete_square_l2100_210078

theorem quadratic_complete_square (r s : ℚ) : 
  (∀ x, 7 * x^2 - 21 * x - 56 = 0 ↔ (x + r)^2 = s) → 
  r + s = 35/4 := by sorry

end quadratic_complete_square_l2100_210078


namespace rectangle_longer_side_l2100_210080

/-- A rectangle with perimeter 60 meters and area 224 square meters has a longer side of 16 meters. -/
theorem rectangle_longer_side (x y : ℝ) (h_perimeter : x + y = 30) (h_area : x * y = 224) 
  (h_x_longer : x ≥ y) : x = 16 := by
  sorry

end rectangle_longer_side_l2100_210080


namespace simplify_expression_l2100_210025

theorem simplify_expression : 2 - (2 / (2 + Real.sqrt 5)) - (2 / (2 - Real.sqrt 5)) = 10 := by
  sorry

end simplify_expression_l2100_210025


namespace intersection_empty_range_l2100_210092

theorem intersection_empty_range (a : ℝ) : 
  let A := {x : ℝ | |x - a| < 1}
  let B := {x : ℝ | 1 < x ∧ x < 5}
  (A ∩ B = ∅) ↔ (a ≤ 0 ∨ a ≥ 6) := by sorry

end intersection_empty_range_l2100_210092


namespace quadratic_always_negative_l2100_210053

theorem quadratic_always_negative (m k : ℝ) :
  (∀ x : ℝ, x^2 - m*x - k + m < 0) ↔ k > m - m^2/4 := by sorry

end quadratic_always_negative_l2100_210053


namespace min_m_value_l2100_210094

/-- The function f(x) = x^2 - 3x --/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The interval [-3, 2] --/
def I : Set ℝ := Set.Icc (-3) 2

/-- The theorem statement --/
theorem min_m_value :
  ∃ (m : ℝ), m = 81/4 ∧ 
  (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ m) ∧
  (∀ (m' : ℝ), (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ m') → m ≤ m') :=
sorry

end min_m_value_l2100_210094


namespace inequality_solution_set_l2100_210039

theorem inequality_solution_set (x : ℝ) : 
  (Set.Iio (-1) ∪ Set.Ioi 3) = {x | (3 - x) / (x + 1) < 0} := by sorry

end inequality_solution_set_l2100_210039


namespace product_inequality_l2100_210038

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end product_inequality_l2100_210038


namespace fathers_age_multiple_l2100_210041

theorem fathers_age_multiple (sons_age : ℕ) (multiple : ℕ) : 
  (44 = multiple * sons_age + 4) →
  (44 + 4 = 2 * (sons_age + 4) + 20) →
  multiple = 4 := by
sorry

end fathers_age_multiple_l2100_210041


namespace fifth_degree_monomial_n_value_l2100_210060

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree (n : ℕ) : ℕ := n + 2 + 1

/-- A monomial 4a^nb^2c is a fifth-degree monomial if its degree is 5 -/
def is_fifth_degree (n : ℕ) : Prop := degree n = 5

theorem fifth_degree_monomial_n_value :
  ∀ n : ℕ, is_fifth_degree n → n = 2 := by
  sorry

end fifth_degree_monomial_n_value_l2100_210060


namespace purely_imaginary_number_l2100_210000

theorem purely_imaginary_number (k : ℝ) : 
  (∃ (z : ℂ), z = (2 * k^2 - 3 * k - 2 : ℝ) + (k^2 - 2 * k : ℝ) * I ∧ z.re = 0 ∧ z.im ≠ 0) → 
  k = -1/2 := by
sorry

end purely_imaginary_number_l2100_210000


namespace certain_number_is_six_l2100_210093

theorem certain_number_is_six : ∃ x : ℝ, (7 * x - 6 - 12 = 4 * x) ∧ (x = 6) := by
  sorry

end certain_number_is_six_l2100_210093


namespace largest_n_for_positive_sum_l2100_210054

/-- Given an arithmetic sequence {a_n} where a_1 = 9 and a_5 = 1,
    the largest natural number n for which the sum of the first n terms (S_n) is positive is 9. -/
theorem largest_n_for_positive_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 9 →
  a 5 = 1 →
  (∀ n, S n = n * (a 1 + a n) / 2) →  -- sum formula for arithmetic sequence
  (∀ m : ℕ, m > 9 → S m ≤ 0) ∧ S 9 > 0 := by
sorry


end largest_n_for_positive_sum_l2100_210054


namespace intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l2100_210037

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | 1 - m ≤ x ∧ x ≤ 3 * m - 1}

-- Theorem for part 1
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) ∧
  (A ∪ B 3 = {x : ℝ | -3 ≤ x ∧ x ≤ 8}) := by
  sorry

-- Theorem for part 2
theorem intersection_equals_B_iff_m_leq_1 (m : ℝ) :
  A ∩ B m = B m ↔ m ≤ 1 := by
  sorry

end intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l2100_210037


namespace probability_different_suits_l2100_210072

def deck_size : ℕ := 60
def num_suits : ℕ := 4
def cards_per_suit : ℕ := 15

theorem probability_different_suits :
  let prob_diff_suits := (deck_size - cards_per_suit) / (deck_size * (deck_size - 1))
  prob_diff_suits = 45 / 236 := by
  sorry

end probability_different_suits_l2100_210072


namespace room_population_l2100_210068

theorem room_population (P M : ℕ) : 
  (P : ℚ) * (2 / 100) = 1 →  -- 2% of painters are musicians
  (M : ℚ) * (5 / 100) = 1 →  -- 5% of musicians are painters
  P + M - 1 = 69             -- Total people in the room
  := by sorry

end room_population_l2100_210068


namespace equation_solutions_l2100_210088

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^3 + x^2*y + x*y^2 + y^3 = 8*(x^2 + x*y + y^2 + 1)

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) :=
  {(8, -2), (-2, 8), (4 + Real.sqrt 15, 4 - Real.sqrt 15), (4 - Real.sqrt 15, 4 + Real.sqrt 15)}

-- Theorem statement
theorem equation_solutions :
  ∀ x y : ℝ, equation x y ↔ (x, y) ∈ solutions := by
  sorry

end equation_solutions_l2100_210088


namespace quadratic_roots_condition_l2100_210030

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) → m < 1 := by
  sorry

end quadratic_roots_condition_l2100_210030


namespace second_month_sale_l2100_210020

def sales_data : List ℕ := [800, 1000, 700, 800, 900]
def num_months : ℕ := 6
def average_sale : ℕ := 850

theorem second_month_sale :
  ∃ (second_month : ℕ),
    (List.sum sales_data + second_month) / num_months = average_sale ∧
    second_month = 900 := by
  sorry

end second_month_sale_l2100_210020


namespace percentage_passed_both_subjects_l2100_210070

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 30)
  (h2 : failed_english = 42)
  (h3 : failed_both = 28) :
  100 - (failed_hindi + failed_english - failed_both) = 56 := by
sorry

end percentage_passed_both_subjects_l2100_210070


namespace davids_math_marks_l2100_210086

theorem davids_math_marks
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (num_subjects : ℕ)
  (h1 : english_marks = 70)
  (h2 : physics_marks = 80)
  (h3 : chemistry_marks = 63)
  (h4 : biology_marks = 65)
  (h5 : average_marks = 68.2)
  (h6 : num_subjects = 5) :
  ∃ math_marks : ℕ,
    math_marks = 63 ∧
    (english_marks + physics_marks + chemistry_marks + biology_marks + math_marks : ℚ) / num_subjects = average_marks :=
by sorry

end davids_math_marks_l2100_210086


namespace doubled_to_original_ratio_l2100_210069

theorem doubled_to_original_ratio (x : ℝ) : 3 * (2 * x + 9) = 57 → (2 * x) / x = 2 := by
  sorry

end doubled_to_original_ratio_l2100_210069


namespace simplify_and_rationalize_l2100_210099

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l2100_210099


namespace f_inequality_implies_a_range_l2100_210029

open Set Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (log x + 2) / x + a * (x - 1) - 2

def domain : Set ℝ := {x | x ∈ (Set.Ioo 0 1) ∪ (Set.Ioi 1)}

theorem f_inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ domain, (f a x) / (1 - x) < a / x) → a ≥ (1 / 2) := by
  sorry

end f_inequality_implies_a_range_l2100_210029


namespace product_sum_relation_l2100_210047

theorem product_sum_relation (a b c : ℚ) 
  (h1 : a * b * c = 2 * (a + b + c) + 14)
  (h2 : b = 8)
  (h3 : c = 5) :
  (c - a)^2 + b = 8513 / 361 := by sorry

end product_sum_relation_l2100_210047


namespace quadratic_roots_property_l2100_210005

theorem quadratic_roots_property (a : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + a*x₁ - 2 = 0) → 
  (x₂^2 + a*x₂ - 2 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^3 + 22/x₂ = x₂^3 + 22/x₁) →
  (a = 3 ∨ a = -3) :=
by sorry

end quadratic_roots_property_l2100_210005


namespace complete_square_sum_l2100_210075

theorem complete_square_sum (a b c : ℝ) (r s : ℝ) :
  (6 * a^2 - 30 * a - 36 = 0) →
  ((a + r)^2 = s) →
  (6 * a^2 - 30 * a - 36 = 6 * ((a + r)^2 - s)) →
  (r + s = 9.75) := by
  sorry

end complete_square_sum_l2100_210075


namespace complement_of_union_M_N_l2100_210009

def U : Finset ℕ := {1,2,3,4,5,6}
def M : Finset ℕ := {2,3,5}
def N : Finset ℕ := {4,5}

theorem complement_of_union_M_N :
  (U \ (M ∪ N)) = {1,6} := by sorry

end complement_of_union_M_N_l2100_210009


namespace no_valid_coloring_l2100_210013

def Color := Fin 3

theorem no_valid_coloring :
  ¬∃ f : ℕ+ → Color,
    (∀ c : Color, ∃ n : ℕ+, f n = c) ∧
    (∀ a b : ℕ+, f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b) :=
by sorry

end no_valid_coloring_l2100_210013


namespace true_masses_l2100_210006

/-- Represents the uneven lever scale with a linear relationship between left and right sides -/
structure UnevenLeverScale where
  k : ℝ
  b : ℝ
  left_to_right : ℝ → ℝ
  right_to_left : ℝ → ℝ
  hk_pos : k > 0
  hleft_to_right : left_to_right = fun x => k * x + b
  hright_to_left : right_to_left = fun y => (y - b) / k

/-- The equilibrium conditions observed on the uneven lever scale -/
structure EquilibriumConditions (scale : UnevenLeverScale) where
  melon_right : scale.left_to_right 3 = scale.right_to_left 5.5
  melon_left : scale.right_to_left 5.5 = scale.left_to_right 3
  watermelon_right : scale.left_to_right 5 = scale.right_to_left 10
  watermelon_left : scale.right_to_left 10 = scale.left_to_right 5

/-- The theorem stating the true masses of the melon and watermelon -/
theorem true_masses (scale : UnevenLeverScale) (conditions : EquilibriumConditions scale) :
  ∃ (melon_mass watermelon_mass : ℝ),
    melon_mass = 5.5 ∧
    watermelon_mass = 10 ∧
    scale.left_to_right 3 = melon_mass ∧
    scale.right_to_left 5.5 = melon_mass ∧
    scale.left_to_right 5 = watermelon_mass ∧
    scale.right_to_left 10 = watermelon_mass := by
  sorry

end true_masses_l2100_210006


namespace correct_observation_value_l2100_210081

theorem correct_observation_value (n : ℕ) (original_mean corrected_mean wrong_value : ℚ) 
  (h1 : n = 50)
  (h2 : original_mean = 41)
  (h3 : corrected_mean = 41.5)
  (h4 : wrong_value = 23) :
  let original_sum := n * original_mean
  let correct_sum := n * corrected_mean
  let correct_value := correct_sum - (original_sum - wrong_value)
  correct_value = 48 := by sorry

end correct_observation_value_l2100_210081


namespace max_value_of_operation_max_value_achieved_l2100_210017

theorem max_value_of_operation (n : ℕ) : 
  (10 ≤ n ∧ n ≤ 99) → 3 * (300 - 2 * n) ≤ 840 := by
sorry

theorem max_value_achieved : 
  ∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ 3 * (300 - 2 * n) = 840 := by
sorry

end max_value_of_operation_max_value_achieved_l2100_210017


namespace squaredigital_numbers_l2100_210065

/-- Sum of digits of a non-negative integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is squaredigital if it equals the square of the sum of its digits -/
def is_squaredigital (n : ℕ) : Prop := n = (sum_of_digits n)^2

/-- The only squaredigital numbers are 0, 1, and 81 -/
theorem squaredigital_numbers : 
  ∀ n : ℕ, is_squaredigital n ↔ n = 0 ∨ n = 1 ∨ n = 81 := by sorry

end squaredigital_numbers_l2100_210065


namespace factory_production_difference_l2100_210057

/-- Represents the production rate and total products for a machine type -/
structure MachineType where
  rate : ℕ  -- products per minute
  total : ℕ -- total products made

/-- Calculates the difference in products between two machine types -/
def productDifference (a b : MachineType) : ℕ :=
  b.total - a.total

theorem factory_production_difference :
  let machineA : MachineType := { rate := 5, total := 25 }
  let machineB : MachineType := { rate := 8, total := 40 }
  productDifference machineA machineB = 15 := by
  sorry

#eval productDifference { rate := 5, total := 25 } { rate := 8, total := 40 }

end factory_production_difference_l2100_210057


namespace absolute_value_inequality_l2100_210061

theorem absolute_value_inequality (x : ℝ) :
  abs (x - 2) + abs (x - 1) ≥ 5 ↔ x ≤ -1 ∨ x ≥ 4 := by sorry

end absolute_value_inequality_l2100_210061


namespace nonagon_diagonals_l2100_210097

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def nonagon_sides : ℕ := 9

theorem nonagon_diagonals : num_diagonals nonagon_sides = 27 := by
  sorry

end nonagon_diagonals_l2100_210097


namespace unique_solution_for_prime_equation_l2100_210012

theorem unique_solution_for_prime_equation (p q r t n : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p^2 + q*t = (p + t)^n → 
  p^2 + q*r = t^4 → 
  (p = 2 ∧ q = 7 ∧ r = 11 ∧ t = 3 ∧ n = 2) := by
sorry

end unique_solution_for_prime_equation_l2100_210012


namespace negation_equivalence_l2100_210026

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 + 3*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 3*x + 2 ≥ 0) := by
  sorry

end negation_equivalence_l2100_210026


namespace evaluate_expression_l2100_210076

theorem evaluate_expression : (4^4 - 4*(4-2)^4)^4 = 1358954496 := by
  sorry

end evaluate_expression_l2100_210076


namespace cube_product_three_six_l2100_210021

theorem cube_product_three_six : 3^3 * 6^3 = 5832 := by
  sorry

end cube_product_three_six_l2100_210021


namespace fresh_corn_processing_capacity_l2100_210089

/-- The daily processing capacity of fresh corn before technological improvement -/
def daily_capacity : ℕ := 2400

/-- The annual processing capacity before technological improvement -/
def annual_capacity : ℕ := 260000

/-- The improvement factor for daily processing capacity -/
def improvement_factor : ℚ := 13/10

/-- The reduction in processing time after improvement (in days) -/
def time_reduction : ℕ := 25

theorem fresh_corn_processing_capacity :
  daily_capacity = 2400 ∧
  annual_capacity = 260000 ∧
  (annual_capacity : ℚ) / daily_capacity - 
    (annual_capacity : ℚ) / (improvement_factor * daily_capacity) = time_reduction := by
  sorry

#check fresh_corn_processing_capacity

end fresh_corn_processing_capacity_l2100_210089


namespace detergent_calculation_l2100_210014

/-- Calculates the amount of detergent in a solution given the ratio of detergent to water and the amount of water -/
def detergent_amount (detergent_ratio : ℚ) (water_ratio : ℚ) (water_amount : ℚ) : ℚ :=
  (detergent_ratio / water_ratio) * water_amount

theorem detergent_calculation :
  let detergent_ratio : ℚ := 1
  let water_ratio : ℚ := 8
  let water_amount : ℚ := 300
  detergent_amount detergent_ratio water_ratio water_amount = 37.5 := by
  sorry

end detergent_calculation_l2100_210014


namespace bisector_sum_ratio_bound_bisector_sum_ratio_bound_tight_l2100_210043

/-- A triangle with sides a, b, c and angle bisectors l_a, l_b -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  l_a : ℝ
  l_b : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  bisector_formula_a : l_a = (2 * b * c * Real.sqrt ((1 + (b^2 + c^2 - a^2) / (2 * b * c)) / 2)) / (b + c)
  bisector_formula_b : l_b = (2 * a * c * Real.sqrt ((1 + (a^2 + c^2 - b^2) / (2 * a * c)) / 2)) / (a + c)

/-- The main theorem: the ratio of sum of bisectors to sum of sides is at most 4/3 -/
theorem bisector_sum_ratio_bound (t : Triangle) : (t.l_a + t.l_b) / (t.a + t.b) ≤ 4/3 := by
  sorry

/-- The bound 4/3 is tight -/
theorem bisector_sum_ratio_bound_tight : 
  ∀ ε > 0, ∃ t : Triangle, (t.l_a + t.l_b) / (t.a + t.b) > 4/3 - ε := by
  sorry

end bisector_sum_ratio_bound_bisector_sum_ratio_bound_tight_l2100_210043


namespace probability_four_twos_in_five_rolls_l2100_210036

theorem probability_four_twos_in_five_rolls (p : ℝ) (h1 : p = 1 / 6) :
  (5 : ℝ) * p^4 * (1 - p) = 5 / 72 := by
  sorry

end probability_four_twos_in_five_rolls_l2100_210036


namespace power_multiplication_l2100_210083

theorem power_multiplication (x : ℝ) : x^5 * x^3 = x^8 := by
  sorry

end power_multiplication_l2100_210083


namespace parabola_point_distance_l2100_210035

/-- Given a parabola y = -ax²/4 + ax + c and three points on it, 
    prove that if y₁ > y₃ ≥ y₂ and y₂ is the vertex, then |x₁ - x₂| > |x₃ - x₂| -/
theorem parabola_point_distance (a c x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  y₁ = -a * x₁^2 / 4 + a * x₁ + c →
  y₂ = -a * x₂^2 / 4 + a * x₂ + c →
  y₃ = -a * x₃^2 / 4 + a * x₃ + c →
  y₂ = a + c →
  y₁ > y₃ →
  y₃ ≥ y₂ →
  |x₁ - x₂| > |x₃ - x₂| := by
  sorry

end parabola_point_distance_l2100_210035


namespace angle_at_larger_base_l2100_210048

/-- An isosceles trapezoid with a regular triangle on its smaller base -/
structure IsoscelesTrapezoidWithTriangle where
  /-- The smaller base of the trapezoid -/
  smallerBase : ℝ
  /-- The larger base of the trapezoid -/
  largerBase : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The angle at the larger base of the trapezoid -/
  angle : ℝ
  /-- The area of the trapezoid -/
  areaT : ℝ
  /-- The area of the triangle -/
  areaTriangle : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The triangle is regular -/
  isRegular : True
  /-- The height of the triangle equals the height of the trapezoid -/
  heightEqual : True
  /-- The area of the triangle is 5 times less than the area of the trapezoid -/
  areaRelation : areaT = 5 * areaTriangle

/-- The theorem to be proved -/
theorem angle_at_larger_base (t : IsoscelesTrapezoidWithTriangle) :
  t.angle = 30 * π / 180 :=
sorry

end angle_at_larger_base_l2100_210048


namespace platform_length_l2100_210001

/-- Given a train of length 750 m that crosses a platform in 65 seconds
    and a signal pole in 30 seconds, the length of the platform is 875 m. -/
theorem platform_length
  (train_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : train_length = 750)
  (h2 : platform_crossing_time = 65)
  (h3 : pole_crossing_time = 30) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 875 :=
by sorry

end platform_length_l2100_210001


namespace triangle_properties_l2100_210010

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (abc : Triangle) :
  (abc.B.cos = -5/13 ∧ 
   (2 * abc.A.sin) * (2 * abc.C.sin) = abc.B.sin^2 ∧ 
   1/2 * abc.a * abc.c * abc.B.sin = 6/13) →
  (abc.a + abc.c) / 2 = Real.sqrt 221 / 13
  ∧
  (abc.B.cos = -5/13 ∧ 
   abc.C.cos = 4/5 ∧ 
   abc.b * abc.c * abc.A.cos = 14) →
  abc.a = 11/4 := by
sorry

end triangle_properties_l2100_210010


namespace no_intersection_empty_union_equality_iff_l2100_210008

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Theorem 1: There is no value of a such that A ∩ B = ∅
theorem no_intersection_empty (a : ℝ) : (A a) ∩ B ≠ ∅ := by
  sorry

-- Theorem 2: A ∪ B = B if and only if a ∈ (-∞, -4) ∪ (5, ∞)
theorem union_equality_iff (a : ℝ) : (A a) ∪ B = B ↔ a < -4 ∨ a > 5 := by
  sorry

end no_intersection_empty_union_equality_iff_l2100_210008


namespace max_correct_answers_l2100_210073

/-- Represents an exam with a specific scoring system and result. -/
structure Exam where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Represents a possible breakdown of answers in an exam. -/
structure ExamResult where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ

/-- Checks if an ExamResult is valid for a given Exam. -/
def is_valid_result (e : Exam) (r : ExamResult) : Prop :=
  r.correct + r.incorrect + r.unanswered = e.total_questions ∧
  r.correct * e.correct_points + r.incorrect * e.incorrect_points = e.total_score

/-- Theorem: The maximum number of correct answers for the given exam is 33. -/
theorem max_correct_answers (e : Exam) :
  e.total_questions = 60 ∧ e.correct_points = 5 ∧ e.incorrect_points = -1 ∧ e.total_score = 140 →
  (∃ (r : ExamResult), is_valid_result e r ∧
    ∀ (r' : ExamResult), is_valid_result e r' → r'.correct ≤ r.correct) ∧
  (∃ (r : ExamResult), is_valid_result e r ∧ r.correct = 33) :=
by sorry

end max_correct_answers_l2100_210073


namespace max_height_triangle_def_l2100_210042

/-- Triangle DEF with sides a, b, c -/
structure Triangle (a b c : ℝ) where
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The maximum possible height of a table constructed from a triangle -/
def max_table_height (t : Triangle a b c) : ℝ :=
  sorry

theorem max_height_triangle_def (t : Triangle 20 29 35) :
  max_table_height t = 84 * Real.sqrt 2002 / 64 := by
  sorry

end max_height_triangle_def_l2100_210042


namespace portion_filled_in_twenty_minutes_l2100_210066

/-- Represents the portion of a cistern filled by a pipe in a given time. -/
def portion_filled (time : ℝ) : ℝ := sorry

/-- The time it takes to fill a certain portion of the cistern. -/
def fill_time : ℝ := 20

/-- Theorem stating that the portion filled in 20 minutes is 1. -/
theorem portion_filled_in_twenty_minutes :
  portion_filled fill_time = 1 := by sorry

end portion_filled_in_twenty_minutes_l2100_210066


namespace outfit_count_l2100_210004

/-- The number of outfits that can be made with different colored shirts and hats -/
def num_outfits (red_shirts green_shirts blue_shirts : ℕ) 
                (pants : ℕ) 
                (green_hats red_hats blue_hats : ℕ) : ℕ :=
  red_shirts * pants * (green_hats + blue_hats) +
  green_shirts * pants * (red_hats + blue_hats) +
  blue_shirts * pants * (green_hats + red_hats)

/-- Theorem stating the number of outfits under given conditions -/
theorem outfit_count : 
  num_outfits 7 6 5 6 6 7 5 = 1284 :=
sorry

end outfit_count_l2100_210004


namespace conference_handshakes_l2100_210015

/-- The number of handshakes in a conference with multiple companies --/
def num_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating that the number of handshakes for the given scenario is 75 --/
theorem conference_handshakes :
  num_handshakes 3 5 = 75 := by
  sorry

end conference_handshakes_l2100_210015


namespace complex_magnitude_l2100_210032

theorem complex_magnitude (z : ℂ) (h : z * (1 - 2*I) = 3 + I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l2100_210032


namespace coefficient_x5_expansion_l2100_210059

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^5 in the expansion of (1-x^3)(1+x)^10
def coefficient_x5 : ℕ := binomial 10 5 - binomial 10 2

-- Theorem statement
theorem coefficient_x5_expansion :
  coefficient_x5 = 207 := by sorry

end coefficient_x5_expansion_l2100_210059


namespace remaining_cherries_l2100_210002

def initial_cherries : ℕ := 77
def cherries_used : ℕ := 60

theorem remaining_cherries : initial_cherries - cherries_used = 17 := by
  sorry

end remaining_cherries_l2100_210002


namespace point_outside_circle_iff_m_in_range_l2100_210044

/-- A circle in the x-y plane defined by the equation x^2 + y^2 + 2x - m = 0 -/
def Circle (m : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - m = 0}

/-- The point P with coordinates (1,1) -/
def P : ℝ × ℝ := (1, 1)

/-- Predicate to check if a point is outside a circle -/
def IsOutside (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∀ q ∈ c, (p.1 - q.1)^2 + (p.2 - q.2)^2 > 0

theorem point_outside_circle_iff_m_in_range :
  ∀ m : ℝ, IsOutside P (Circle m) ↔ -1 < m ∧ m < 4 :=
sorry

end point_outside_circle_iff_m_in_range_l2100_210044


namespace last_two_digits_of_fraction_l2100_210074

theorem last_two_digits_of_fraction (n : ℕ) : n = 50 * 52 * 54 * 56 * 58 * 60 →
  n / 8000 ≡ 22 [ZMOD 100] :=
by
  sorry

end last_two_digits_of_fraction_l2100_210074


namespace max_area_rectangular_garden_l2100_210087

/-- The maximum area of a rectangular garden enclosed by a fence of length 36m is 81 m² -/
theorem max_area_rectangular_garden : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2*(x + y) = 36 → x*y ≤ 81 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*(x + y) = 36 ∧ x*y = 81 :=
by sorry

end max_area_rectangular_garden_l2100_210087


namespace smallest_constant_term_l2100_210084

theorem smallest_constant_term (a b c d e : ℤ) : 
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -2 ∨ x = 5 ∨ x = 9 ∨ x = -1/2) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e' = 0 ↔ 
      x = -2 ∨ x = 5 ∨ x = 9 ∨ x = -1/2) → 
    e ≤ e') →
  e = 90 :=
by sorry

end smallest_constant_term_l2100_210084


namespace shoe_repair_cost_l2100_210085

theorem shoe_repair_cost (new_shoe_cost : ℝ) (new_shoe_lifespan : ℝ) (repaired_shoe_lifespan : ℝ) (cost_difference_percentage : ℝ) :
  new_shoe_cost = 30 →
  new_shoe_lifespan = 2 →
  repaired_shoe_lifespan = 1 →
  cost_difference_percentage = 42.857142857142854 →
  ∃ repair_cost : ℝ,
    repair_cost = 10.5 ∧
    (new_shoe_cost / new_shoe_lifespan) = repair_cost * (1 + cost_difference_percentage / 100) :=
by sorry

end shoe_repair_cost_l2100_210085


namespace three_times_relation_l2100_210007

/-- Given four numbers M₁, M₂, M₃, and M₄, prove that M₄ = 3M₂ -/
theorem three_times_relation (M₁ M₂ M₃ M₄ : ℝ) 
  (hM₁ : M₁ = 2.02e-6)
  (hM₂ : M₂ = 0.0000202)
  (hM₃ : M₃ = 0.00000202)
  (hM₄ : M₄ = 6.06e-5) :
  M₄ = 3 * M₂ := by
  sorry

end three_times_relation_l2100_210007


namespace false_proposition_l2100_210067

-- Define the lines
def line1 : ℝ → ℝ → Prop := λ x y => 6*x + 2*y - 1 = 0
def line2 : ℝ → ℝ → Prop := λ x y => y = 5 - 3*x
def line3 : ℝ → ℝ → Prop := λ x y => 2*x + 6*y - 4 = 0

-- Define the propositions
def p : Prop := ∀ x y, line1 x y ↔ line2 x y
def q : Prop := ∀ x y, line1 x y → line3 x y

-- Theorem statement
theorem false_proposition : ¬((¬p) ∧ q) := by
  sorry

end false_proposition_l2100_210067


namespace flowers_after_one_month_l2100_210023

/-- Represents the number of flowers in Mark's garden -/
structure GardenFlowers where
  yellow : ℕ
  purple : ℕ
  green : ℕ
  red : ℕ

/-- Calculates the number of flowers after one month -/
def flowersAfterOneMonth (initial : GardenFlowers) : ℕ :=
  let yellowAfter := initial.yellow + (initial.yellow / 2)
  let purpleAfter := initial.purple * 2
  let greenAfter := initial.green - (initial.green / 5)
  let redAfter := initial.red + (initial.red * 4 / 5)
  yellowAfter + purpleAfter + greenAfter + redAfter

/-- Theorem stating the number of flowers after one month -/
theorem flowers_after_one_month :
  ∃ (initial : GardenFlowers),
    initial.yellow = 10 ∧
    initial.purple = initial.yellow + (initial.yellow * 4 / 5) ∧
    initial.green = (initial.yellow + initial.purple) / 4 ∧
    initial.red = ((initial.yellow + initial.purple + initial.green) * 35) / 100 ∧
    flowersAfterOneMonth initial = 77 :=
  sorry

end flowers_after_one_month_l2100_210023


namespace bank_deposit_problem_l2100_210018

theorem bank_deposit_problem (P : ℝ) : 
  P > 0 →
  (0.15 * P * 5 - 0.15 * P * 3.5 = 144) →
  P = 640 := by
sorry

end bank_deposit_problem_l2100_210018


namespace diophantine_equation_solution_l2100_210062

theorem diophantine_equation_solution (x y : ℤ) :
  5 * x - 7 * y = 3 →
  ∃ t : ℤ, x = 7 * t - 12 ∧ y = 5 * t - 9 :=
by sorry

end diophantine_equation_solution_l2100_210062


namespace factory_sampling_is_systematic_l2100_210034

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a factory with a conveyor belt and sampling process -/
structure Factory where
  sampleInterval : ℕ  -- Time interval between samples in minutes
  sampleLocation : String  -- Description of the sample location

/-- Determines the sampling method based on the factory's sampling process -/
def determineSamplingMethod (f : Factory) : SamplingMethod :=
  sorry

/-- Theorem stating that the described sampling method is systematic sampling -/
theorem factory_sampling_is_systematic (f : Factory) 
  (h1 : f.sampleInterval = 10)
  (h2 : f.sampleLocation = "specific location on the conveyor belt") :
  determineSamplingMethod f = SamplingMethod.Systematic :=
sorry

end factory_sampling_is_systematic_l2100_210034


namespace optimal_transport_solution_l2100_210031

/-- Represents the optimal solution for transporting cargo -/
structure CargoTransport where
  large_trucks : ℕ
  small_trucks : ℕ
  total_fuel : ℕ

/-- Finds the optimal cargo transport solution -/
def find_optimal_transport (total_cargo : ℕ) (large_capacity : ℕ) (small_capacity : ℕ)
  (large_fuel : ℕ) (small_fuel : ℕ) : CargoTransport :=
  sorry

/-- Theorem stating the optimal solution for the given problem -/
theorem optimal_transport_solution :
  let total_cargo : ℕ := 89
  let large_capacity : ℕ := 7
  let small_capacity : ℕ := 4
  let large_fuel : ℕ := 14
  let small_fuel : ℕ := 9
  let solution := find_optimal_transport total_cargo large_capacity small_capacity large_fuel small_fuel
  solution.total_fuel = 181 ∧
  solution.large_trucks * large_capacity + solution.small_trucks * small_capacity ≥ total_cargo :=
by sorry

end optimal_transport_solution_l2100_210031


namespace pq_satisfies_stewarts_theorem_l2100_210016

/-- Triangle DEF with given side lengths and points P and Q -/
structure TriangleDEF where
  -- Side lengths
  DE : ℝ
  EF : ℝ
  DF : ℝ
  -- P is the midpoint of DE
  P : ℝ × ℝ
  -- Q is the foot of the perpendicular from D to EF
  Q : ℝ × ℝ
  -- Conditions
  de_length : DE = 17
  ef_length : EF = 18
  df_length : DF = 19
  p_midpoint : P.1 = DE / 2
  q_perpendicular : sorry -- This would require more geometric setup

/-- The length of PQ satisfies Stewart's Theorem -/
theorem pq_satisfies_stewarts_theorem (t : TriangleDEF) : 
  ∃ (PQ : ℝ), t.DE * (t.DE / 2)^2 + t.DE * PQ^2 = t.DF * (t.DE / 2) * t.DF + t.EF * (t.DE / 2) * t.EF := by
  sorry

#check pq_satisfies_stewarts_theorem

end pq_satisfies_stewarts_theorem_l2100_210016


namespace exists_two_students_with_not_less_scores_l2100_210056

/-- Represents a student's scores for three problems -/
structure StudentScores where
  problem1 : Fin 8
  problem2 : Fin 8
  problem3 : Fin 8

/-- Checks if one student's scores are not less than another's -/
def scoresNotLessThan (a b : StudentScores) : Prop :=
  a.problem1 ≥ b.problem1 ∧ a.problem2 ≥ b.problem2 ∧ a.problem3 ≥ b.problem3

/-- The main theorem to be proved -/
theorem exists_two_students_with_not_less_scores 
  (students : Fin 249 → StudentScores) : 
  ∃ (i j : Fin 249), i ≠ j ∧ scoresNotLessThan (students i) (students j) := by
  sorry

end exists_two_students_with_not_less_scores_l2100_210056


namespace committee_selection_ways_l2100_210063

/-- The number of ways to choose a k-person committee from a group of n people -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the club -/
def club_size : ℕ := 11

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- Theorem stating that the number of ways to choose a 5-person committee from a club of 11 people is 462 -/
theorem committee_selection_ways : choose club_size committee_size = 462 := by
  sorry

end committee_selection_ways_l2100_210063
