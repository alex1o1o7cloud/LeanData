import Mathlib

namespace arithmetic_square_root_of_one_fourth_l3551_355177

theorem arithmetic_square_root_of_one_fourth :
  let x : ℚ := 1/2
  (x * x = 1/4) ∧ (∀ y : ℚ, y * y = 1/4 → y = x ∨ y = -x) :=
by sorry

end arithmetic_square_root_of_one_fourth_l3551_355177


namespace find_number_l3551_355133

theorem find_number (x : ℕ) : 102 * 102 + x * x = 19808 → x = 97 := by
  sorry

end find_number_l3551_355133


namespace complex_sum_of_powers_l3551_355135

theorem complex_sum_of_powers (i : ℂ) : i^2 = -1 → i + i^2 + i^3 = -1 := by
  sorry

end complex_sum_of_powers_l3551_355135


namespace f_difference_at_3_and_neg_3_l3551_355109

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + 7*x

-- Theorem statement
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 582 := by
  sorry

end f_difference_at_3_and_neg_3_l3551_355109


namespace isosceles_triangle_perimeter_l3551_355158

/-- Given a quadratic equation x^2 - 9x + 18 = 0, if its roots represent the base and legs
    of an isosceles triangle, then the perimeter of the triangle is 15. -/
theorem isosceles_triangle_perimeter (x : ℝ) : 
  x^2 - 9*x + 18 = 0 →
  ∃ (base leg : ℝ), 
    (x = base ∨ x = leg) ∧ 
    (base > 0 ∧ leg > 0) ∧
    (2*leg > base) ∧
    (base + 2*leg = 15) :=
by sorry

end isosceles_triangle_perimeter_l3551_355158


namespace prob_blue_face_four_blue_two_red_l3551_355116

/-- A cube with blue and red faces -/
structure ColoredCube where
  blue_faces : ℕ
  red_faces : ℕ

/-- The probability of rolling a blue face on a colored cube -/
def prob_blue_face (cube : ColoredCube) : ℚ :=
  cube.blue_faces / (cube.blue_faces + cube.red_faces)

/-- Theorem: The probability of rolling a blue face on a cube with 4 blue faces and 2 red faces is 2/3 -/
theorem prob_blue_face_four_blue_two_red :
  prob_blue_face ⟨4, 2⟩ = 2/3 := by
  sorry

end prob_blue_face_four_blue_two_red_l3551_355116


namespace f_satisfies_properties_l3551_355184

def f (x : ℝ) : ℝ := (x - 2)^2

-- Property 1: f(x+2) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f (-x + 2)

-- Property 2: f(x) is decreasing on (-∞, 2) and increasing on (2, +∞)
def is_decreasing_then_increasing (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x < y ∧ y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x ∧ x < y → f x < f y)

theorem f_satisfies_properties : 
  is_even_shifted f ∧ is_decreasing_then_increasing f :=
sorry

end f_satisfies_properties_l3551_355184


namespace intersection_A_complement_B_l3551_355117

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x^2 + 2*x < 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {-2, 0, 1, 2} := by
  sorry

end intersection_A_complement_B_l3551_355117


namespace president_and_committee_selection_l3551_355174

theorem president_and_committee_selection (n : ℕ) (k : ℕ) : 
  n = 10 → k = 3 → n * (Nat.choose (n - 1) k) = 840 := by
  sorry

end president_and_committee_selection_l3551_355174


namespace problem_solution_l3551_355141

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x : ℝ | a + 1 ≤ x ∧ x ≤ 2 * a - 1}

theorem problem_solution :
  (∀ x : ℝ, x ∈ (M ∪ N (7/2)) ↔ -2 ≤ x ∧ x ≤ 6) ∧
  (∀ x : ℝ, x ∈ ((Set.univ \ M) ∩ N (7/2)) ↔ 5 < x ∧ x ≤ 6) ∧
  (∀ a : ℝ, M ⊇ N a ↔ a ≤ 3) :=
by sorry

end problem_solution_l3551_355141


namespace two_sevens_numeral_l3551_355165

/-- Given two sevens in a numeral with a difference of 69930 between their place values,
    prove that the numeral is 7700070. -/
theorem two_sevens_numeral (A B : ℕ) : 
  A - B = 69930 →
  A = 10 * B →
  A = 77700 ∧ B = 7770 ∧ 7700070 = 7 * A + 7 * B :=
by sorry

end two_sevens_numeral_l3551_355165


namespace trig_ratio_simplification_l3551_355130

theorem trig_ratio_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 := by
  sorry

end trig_ratio_simplification_l3551_355130


namespace arithmetic_expression_equality_l3551_355154

theorem arithmetic_expression_equality : 70 + (105 / 15) + (19 * 11) - 250 - (360 / 12) = 6 := by
  sorry

end arithmetic_expression_equality_l3551_355154


namespace solution_of_linear_equation_l3551_355138

theorem solution_of_linear_equation (a : ℚ) : 
  (∃ x y : ℚ, x = 2 ∧ y = 2 ∧ a * x + y = 5) → a = 3/2 := by
  sorry

end solution_of_linear_equation_l3551_355138


namespace no_maximum_on_interval_l3551_355143

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

-- Define the property of being an even function
def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Theorem statement
theorem no_maximum_on_interval (m : ℝ) :
  is_even_function (f m) →
  ¬∃ (y : ℝ), ∀ x ∈ Set.Ioo (-2 : ℝ) (-1), f m x ≤ y :=
by sorry

end no_maximum_on_interval_l3551_355143


namespace no_eight_consecutive_odd_exponent_primes_l3551_355113

theorem no_eight_consecutive_odd_exponent_primes :
  ∀ n : ℕ, ∃ k : ℕ, k ∈ Finset.range 8 ∧
  ∃ p : ℕ, Prime p ∧ ∃ m : ℕ, m > 0 ∧ 2 ∣ m ∧ p ^ m ∣ (n + k) := by
  sorry

end no_eight_consecutive_odd_exponent_primes_l3551_355113


namespace geometric_relations_l3551_355183

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (perpendicular : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (contains : Plane → Point → Prop)
variable (on_line : Point → Line → Prop)
variable (plane_through_point_perp_to_line : Point → Line → Plane)
variable (line_perp_to_plane : Point → Plane → Line)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem geometric_relations 
  (α β : Plane) (l : Line) (P : Point) 
  (h1 : perpendicular α β)
  (h2 : intersection α β = l)
  (h3 : contains α P)
  (h4 : ¬ on_line P l) :
  (perpendicular (plane_through_point_perp_to_line P l) β) ∧ 
  (parallel (line_perp_to_plane P α) β) ∧
  (line_in_plane (line_perp_to_plane P β) α) :=
sorry

end geometric_relations_l3551_355183


namespace equation_solutions_l3551_355178

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 - (x - 2) = 0 ↔ x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, x^2 - x = x + 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) := by
  sorry

end equation_solutions_l3551_355178


namespace divisibility_property_l3551_355157

theorem divisibility_property (m n p : ℕ) (h_prime : Nat.Prime p) 
  (h_order : m < n ∧ n < p) (h_div_m : p ∣ m^2 + 1) (h_div_n : p ∣ n^2 + 1) : 
  p ∣ m * n - 1 := by
sorry

end divisibility_property_l3551_355157


namespace one_and_two_thirds_of_x_is_45_l3551_355196

theorem one_and_two_thirds_of_x_is_45 : ∃ x : ℝ, (5/3) * x = 45 ∧ x = 27 := by
  sorry

end one_and_two_thirds_of_x_is_45_l3551_355196


namespace calculate_expression_l3551_355149

theorem calculate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -a - b^3 + a*b^2 = 59 := by
  sorry

end calculate_expression_l3551_355149


namespace partial_fraction_decomposition_constant_l3551_355111

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 11*x + 15

-- Define the partial fraction decomposition
def pfd (x A B C : ℝ) : Prop :=
  1 / p x = A / (x - 5) + B / (x + 3) + C / ((x + 3)^2)

-- State the theorem
theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x, pfd x A B C) → (∀ x, p x = (x - 5) * (x + 3)^2) → A = 1/64 := by
  sorry

end partial_fraction_decomposition_constant_l3551_355111


namespace min_value_P_l3551_355171

theorem min_value_P (a b : ℝ) (h : ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (∀ t : ℝ, a * t^3 - t^2 + b * t - 1 = 0 ↔ t = x ∨ t = y ∨ t = z)) :
  ∀ P : ℝ, P = (5 * a^2 - 3 * a * b + 2) / (a^2 * (b - a)) → P ≥ 4 * Real.sqrt 3 :=
sorry

end min_value_P_l3551_355171


namespace armstrong_made_quote_l3551_355166

-- Define the type for astronauts
inductive Astronaut : Type
| Apollo : Astronaut
| MichaelCollins : Astronaut
| Armstrong : Astronaut
| Aldrin : Astronaut

-- Define the famous quote
def famous_quote : String := "That's one small step for man, one giant leap for mankind."

-- Define the property of making the quote on the Moon
def made_quote_on_moon (a : Astronaut) : Prop := 
  a = Astronaut.Armstrong ∧ ∃ (quote : String), quote = famous_quote

-- Theorem stating that Armstrong made the famous quote on the Moon
theorem armstrong_made_quote : 
  ∃ (a : Astronaut), made_quote_on_moon a :=
sorry

end armstrong_made_quote_l3551_355166


namespace square_area_ratio_l3551_355110

theorem square_area_ratio (r : ℝ) (hr : r > 0) :
  let s1 := Real.sqrt ((4 / 5) * r ^ 2)
  let s2 := Real.sqrt (2 * r ^ 2)
  (s1 ^ 2) / (s2 ^ 2) = 2 / 5 := by
  sorry

end square_area_ratio_l3551_355110


namespace parabola_vertex_l3551_355104

/-- The vertex coordinates of the parabola y = x^2 - 6x + 1 are (3, -8) -/
theorem parabola_vertex : 
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 1
  ∃ a b : ℝ, a = 3 ∧ b = -8 ∧ ∀ x : ℝ, f x = (x - a)^2 + b :=
by sorry

end parabola_vertex_l3551_355104


namespace sqrt_18_times_sqrt_32_l3551_355192

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end sqrt_18_times_sqrt_32_l3551_355192


namespace well_digging_payment_l3551_355137

/-- The total payment for two workers digging a well over three days -/
def total_payment (hourly_rate : ℕ) (day1_hours day2_hours day3_hours : ℕ) (num_workers : ℕ) : ℕ :=
  hourly_rate * (day1_hours + day2_hours + day3_hours) * num_workers

/-- Theorem stating that the total payment for the given scenario is $660 -/
theorem well_digging_payment :
  total_payment 10 10 8 15 2 = 660 := by
  sorry

end well_digging_payment_l3551_355137


namespace like_terms_difference_l3551_355127

def are_like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∃ (c d : ℚ) (m n : ℕ), ∀ (x y : ℕ), a x y = c * x^m * y^3 ∧ b x y = d * x^4 * y^n

theorem like_terms_difference (m n : ℕ) :
  are_like_terms (λ x y => -3 * x^m * y^3) (λ x y => 2 * x^4 * y^n) → m - n = 1 := by
  sorry

end like_terms_difference_l3551_355127


namespace negative_x_sqrt_squared_diff_l3551_355108

theorem negative_x_sqrt_squared_diff (x : ℝ) (h : x < 0) : x - Real.sqrt ((x - 1)^2) = 2*x - 1 := by
  sorry

end negative_x_sqrt_squared_diff_l3551_355108


namespace smallest_unstuck_perimeter_l3551_355181

/-- A rectangle inscribed in a larger rectangle. -/
structure InscribedRectangle where
  outer_width : ℝ
  outer_height : ℝ
  inner_width : ℝ
  inner_height : ℝ
  is_inscribed : inner_width ≤ outer_width ∧ inner_height ≤ outer_height

/-- An unstuck inscribed rectangle can be rotated slightly within the larger rectangle. -/
def is_unstuck (r : InscribedRectangle) : Prop := sorry

/-- The perimeter of a rectangle. -/
def perimeter (width height : ℝ) : ℝ := 2 * (width + height)

/-- The theorem to be proved. -/
theorem smallest_unstuck_perimeter :
  ∃ (r : InscribedRectangle),
    r.outer_width = 8 ∧
    r.outer_height = 6 ∧
    is_unstuck r ∧
    (∀ (s : InscribedRectangle),
      s.outer_width = 8 ∧
      s.outer_height = 6 ∧
      is_unstuck s →
      perimeter r.inner_width r.inner_height ≤ perimeter s.inner_width s.inner_height) ∧
    perimeter r.inner_width r.inner_height = Real.sqrt 448 := by sorry

end smallest_unstuck_perimeter_l3551_355181


namespace complex_difference_magnitude_l3551_355173

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 2)
  (h3 : z₁ + z₂ = 1 + Complex.I * Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 2 * Real.sqrt 3 := by
  sorry

end complex_difference_magnitude_l3551_355173


namespace readers_overlap_l3551_355123

theorem readers_overlap (total : ℕ) (sci_fi : ℕ) (literary : ℕ) (h1 : total = 250) (h2 : sci_fi = 180) (h3 : literary = 88) :
  sci_fi + literary - total = 18 := by
  sorry

end readers_overlap_l3551_355123


namespace total_balls_l3551_355125

/-- Given the number of basketballs, volleyballs, and soccer balls in a school,
    prove that the total number of balls is 94. -/
theorem total_balls (b v s : ℕ) : 
  b = 32 →
  b = v + 5 →
  b = s - 3 →
  b + v + s = 94 := by
  sorry

end total_balls_l3551_355125


namespace no_solution_iff_m_less_than_neg_two_l3551_355128

theorem no_solution_iff_m_less_than_neg_two (m : ℝ) :
  (∀ x : ℝ, ¬(x - m ≤ 2*m + 3 ∧ (x - 1)/2 ≥ m)) ↔ m < -2 :=
by sorry

end no_solution_iff_m_less_than_neg_two_l3551_355128


namespace sqrt_3_irrational_l3551_355121

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_3_irrational_l3551_355121


namespace find_r_l3551_355163

theorem find_r (a b c r : ℝ) 
  (h1 : a * (b - c) / (b * (c - a)) = r)
  (h2 : b * (c - a) / (c * (b - a)) = r)
  (h3 : r > 0) :
  r = (Real.sqrt 5 - 1) / 2 := by
  sorry

end find_r_l3551_355163


namespace parabola_point_distance_to_focus_l3551_355140

/-- Given a parabola y = 4x² and a point M(x, y) on the parabola,
    if the distance from M to the focus (0, 1/16) is 1,
    then the y-coordinate of M is 15/16 -/
theorem parabola_point_distance_to_focus (x y : ℝ) :
  y = 4 * x^2 →
  (x - 0)^2 + (y - 1/16)^2 = 1 →
  y = 15/16 := by
sorry

end parabola_point_distance_to_focus_l3551_355140


namespace pure_imaginary_fraction_l3551_355197

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end pure_imaginary_fraction_l3551_355197


namespace arithmetic_mean_fraction_l3551_355191

theorem arithmetic_mean_fraction (x b : ℝ) (h : x ≠ 0) :
  ((x + b) / x + (x - 2 * b) / x) / 2 = 1 - b / (2 * x) := by
sorry

end arithmetic_mean_fraction_l3551_355191


namespace product_of_terms_l3551_355153

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ r : ℝ), ∀ n, b n = b₁ * r^(n - 1)

/-- Main theorem -/
theorem product_of_terms (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a n ≠ 0) →
  2 * (a 2) - (a 7)^2 + 2 * (a 12) = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 3 * b 11 = 16 := by
  sorry

end product_of_terms_l3551_355153


namespace revenue_condition_l3551_355164

def initial_price : ℝ := 50
def initial_sales : ℝ := 300
def revenue_threshold : ℝ := 15950

def monthly_revenue (x : ℝ) : ℝ := (initial_price - x) * (initial_sales + 10 * x)

theorem revenue_condition (x : ℝ) :
  monthly_revenue x ≥ revenue_threshold ↔ (x = 9 ∨ x = 11) :=
sorry

end revenue_condition_l3551_355164


namespace sqrt_x_plus_reciprocal_l3551_355172

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_x_plus_reciprocal_l3551_355172


namespace expressions_same_type_l3551_355120

/-- Two expressions are of the same type if they have the same variables with the same exponents -/
def same_type (e1 e2 : ℕ → ℕ → ℕ → ℚ) : Prop :=
  ∀ a b c : ℕ, ∃ k1 k2 : ℚ, e1 a b c = k1 * a * b^3 * c ∧ e2 a b c = k2 * a * b^3 * c

/-- The original expression -/
def original (a b c : ℕ) : ℚ := -↑a * ↑b^3 * ↑c

/-- The expression to compare -/
def to_compare (a b c : ℕ) : ℚ := (1/3) * ↑a * ↑c * ↑b^3

/-- Theorem stating that the two expressions are of the same type -/
theorem expressions_same_type : same_type original to_compare := by
  sorry

end expressions_same_type_l3551_355120


namespace find_number_l3551_355122

theorem find_number : ∃ x : ℝ, x = 800 ∧ 0.4 * x = 0.2 * 650 + 190 := by
  sorry

end find_number_l3551_355122


namespace calculate_interest_rate_loan_interest_rate_proof_l3551_355105

/-- Calculates the rate of interest for a loan with simple interest -/
theorem calculate_interest_rate (principal : ℝ) (interest_paid : ℝ) : ℝ :=
  let rate_squared := (100 * interest_paid) / (principal)
  Real.sqrt rate_squared

/-- Proves that the rate of interest for the given loan conditions is approximately 8.888% -/
theorem loan_interest_rate_proof 
  (principal : ℝ) 
  (interest_paid : ℝ) 
  (h1 : principal = 800) 
  (h2 : interest_paid = 632) : 
  ∃ (ε : ℝ), ε > 0 ∧ |calculate_interest_rate principal interest_paid - 8.888| < ε :=
sorry

end calculate_interest_rate_loan_interest_rate_proof_l3551_355105


namespace binomial_and_permutation_60_3_l3551_355131

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the permutation
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Theorem statement
theorem binomial_and_permutation_60_3 :
  binomial 60 3 = 34220 ∧ permutation 60 3 = 205320 :=
by sorry

end binomial_and_permutation_60_3_l3551_355131


namespace coordinates_wrt_symmetric_point_l3551_355176

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry about y-axis
def symmetricAboutYAxis (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = p.y

-- Theorem statement
theorem coordinates_wrt_symmetric_point (A B : Point2D) :
  A.x = -5 ∧ A.y = 2 ∧ symmetricAboutYAxis A B →
  (A.x - B.x = 5 ∧ A.y - B.y = 0) := by
  sorry

end coordinates_wrt_symmetric_point_l3551_355176


namespace binary_10010_is_18_l3551_355107

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10010_is_18 :
  binary_to_decimal [false, true, false, false, true] = 18 := by
  sorry

end binary_10010_is_18_l3551_355107


namespace simplify_expression_l3551_355112

theorem simplify_expression (m : ℝ) (h1 : m ≠ -1) (h2 : m ≠ -2) :
  ((4 * m + 5) / (m + 1) + m - 1) / ((m + 2) / (m + 1)) = m + 2 := by
  sorry

end simplify_expression_l3551_355112


namespace chewing_gum_price_l3551_355119

def currency_denominations : List Nat := [1, 5, 10, 20, 50, 100]

def is_valid_payment (price : Nat) (payment1 payment2 : Nat) : Prop :=
  payment1 > price ∧ payment2 > price ∧
  ∃ (exchange : Nat), exchange ≤ payment1 ∧ exchange ≤ payment2 ∧
    payment1 - exchange + (payment2 - price) = price ∧
    payment2 - (payment2 - price) + exchange = price

def exists_valid_payments (price : Nat) : Prop :=
  ∃ (payment1 payment2 : Nat),
    payment1 ∈ currency_denominations ∧
    payment2 ∈ currency_denominations ∧
    is_valid_payment price payment1 payment2

theorem chewing_gum_price :
  ¬ exists_valid_payments 2 ∧
  ¬ exists_valid_payments 6 ∧
  ¬ exists_valid_payments 7 ∧
  exists_valid_payments 8 :=
by sorry

end chewing_gum_price_l3551_355119


namespace greatest_integer_difference_l3551_355188

theorem greatest_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  (⌊y⌋ - ⌈x⌉ : ℤ) ≤ 2 ∧ ∃ (x' y' : ℝ), 4 < x' ∧ x' < 8 ∧ 8 < y' ∧ y' < 12 ∧ (⌊y'⌋ - ⌈x'⌉ : ℤ) = 2 :=
by sorry

end greatest_integer_difference_l3551_355188


namespace divisibility_by_1946_l3551_355146

theorem divisibility_by_1946 (n : ℕ) (hn : n ≤ 1945) :
  ∃ k : ℤ, 1492^n - 1770^n - 1863^n + 2141^n = 1946 * k := by
  sorry


end divisibility_by_1946_l3551_355146


namespace f_3_equals_7_l3551_355199

-- Define the function f
def f : ℝ → ℝ := fun x => 2*x + 1

-- State the theorem
theorem f_3_equals_7 : f 3 = 7 := by
  sorry

end f_3_equals_7_l3551_355199


namespace arithmetic_computation_l3551_355168

theorem arithmetic_computation : -9 * 5 - (-7 * -2) + (-11 * -6) = 7 := by
  sorry

end arithmetic_computation_l3551_355168


namespace range_of_m_l3551_355124

-- Define set A
def A : Set ℝ := {x | x^2 + 3*x - 10 ≤ 0}

-- Define set B (parametrized by m)
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem statement
theorem range_of_m (m : ℝ) : B m ⊆ A → m < 2 := by
  sorry

end range_of_m_l3551_355124


namespace gcd_689_1021_l3551_355156

theorem gcd_689_1021 : Nat.gcd 689 1021 = 1 := by
  sorry

end gcd_689_1021_l3551_355156


namespace point_on_terminal_side_l3551_355145

/-- Given a point P(x,6) on the terminal side of angle θ with cos θ = -4/5, prove that x = -8 -/
theorem point_on_terminal_side (x : ℝ) (θ : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (x, 6) ∧ P.1 = x * Real.cos θ ∧ P.2 = x * Real.sin θ) →
  Real.cos θ = -4/5 →
  x = -8 :=
by sorry

end point_on_terminal_side_l3551_355145


namespace equation_solution_l3551_355194

theorem equation_solution : ∃ x : ℝ, 0.05 * x + 0.07 * (25 + x) = 15.1 ∧ x = 111.25 := by
  sorry

end equation_solution_l3551_355194


namespace f_zero_eq_five_l3551_355189

/-- Given a function f such that f(x-2) = 2^x - x + 3 for all x, prove that f(0) = 5 -/
theorem f_zero_eq_five (f : ℝ → ℝ) (h : ∀ x, f (x - 2) = 2^x - x + 3) : f 0 = 5 := by
  sorry

end f_zero_eq_five_l3551_355189


namespace cost_of_16_pencils_10_notebooks_l3551_355147

/-- The cost of pencils and notebooks given specific quantities -/
def cost_of_items (pencil_price notebook_price : ℚ) (num_pencils num_notebooks : ℕ) : ℚ :=
  pencil_price * num_pencils + notebook_price * num_notebooks

/-- The theorem stating the cost of 16 pencils and 10 notebooks -/
theorem cost_of_16_pencils_10_notebooks :
  ∀ (pencil_price notebook_price : ℚ),
    cost_of_items pencil_price notebook_price 7 8 = 415/100 →
    cost_of_items pencil_price notebook_price 5 3 = 177/100 →
    cost_of_items pencil_price notebook_price 16 10 = 584/100 := by
  sorry

end cost_of_16_pencils_10_notebooks_l3551_355147


namespace choose_three_from_eight_l3551_355169

theorem choose_three_from_eight :
  Nat.choose 8 3 = 56 := by sorry

end choose_three_from_eight_l3551_355169


namespace adjacent_chair_subsets_theorem_l3551_355102

/-- The number of subsets containing at least three adjacent chairs in a circular arrangement of 12 chairs -/
def adjacent_chair_subsets : ℕ := 1634

/-- The number of chairs arranged in a circle -/
def num_chairs : ℕ := 12

/-- A function that calculates the number of subsets containing at least three adjacent chairs -/
def calculate_subsets (n : ℕ) : ℕ := sorry

theorem adjacent_chair_subsets_theorem :
  calculate_subsets num_chairs = adjacent_chair_subsets :=
by sorry

end adjacent_chair_subsets_theorem_l3551_355102


namespace concentric_circles_area_ratio_l3551_355129

theorem concentric_circles_area_ratio :
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > 0 → r₂ = 3 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 := by
  sorry

end concentric_circles_area_ratio_l3551_355129


namespace cos_difference_special_l3551_355185

theorem cos_difference_special (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = -3/8 := by
  sorry

end cos_difference_special_l3551_355185


namespace apple_distribution_l3551_355198

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- Theorem: There are 190 ways to distribute 30 apples among 3 people, with each person receiving at least 4 apples -/
theorem apple_distribution : distribution_ways 30 3 4 = 190 := by
  sorry

end apple_distribution_l3551_355198


namespace inscribed_cylinder_radius_l3551_355152

/-- 
Given a right circular cone with diameter 16 units and altitude 20 units, 
and an inscribed right circular cylinder with height equal to its diameter 
and coinciding axis with the cone, the radius of the cylinder is 40/9 units.
-/
theorem inscribed_cylinder_radius 
  (cone_diameter : ℝ) 
  (cone_altitude : ℝ) 
  (cylinder_radius : ℝ) :
  cone_diameter = 16 →
  cone_altitude = 20 →
  cylinder_radius * 2 = cylinder_radius * 2 →  -- Height equals diameter
  (cone_altitude - cylinder_radius * 2) / cylinder_radius = 5 / 2 →  -- Similar triangles ratio
  cylinder_radius = 40 / 9 := by
  sorry

end inscribed_cylinder_radius_l3551_355152


namespace train_speed_calculation_l3551_355187

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 145 →
  bridge_length = 230 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l3551_355187


namespace flash_catches_ace_l3551_355139

/-- The distance Flash must run to catch Ace -/
def flashDistance (x v c y : ℝ) : ℝ := 2 * y

theorem flash_catches_ace (x v c y : ℝ) 
  (hx : x > 1) 
  (hc : c > 0) : 
  flashDistance x v c y = 2 * y := by
  sorry

#check flash_catches_ace

end flash_catches_ace_l3551_355139


namespace total_fish_count_l3551_355162

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1 : num_fishbowls = 261) 
  (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := by
  sorry

end total_fish_count_l3551_355162


namespace subset_condition_nonempty_intersection_l3551_355118

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 + 12*x - 20 > 0}
def B (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Theorem for part (1)
theorem subset_condition (a : ℝ) : B a ⊆ A → a ∈ Set.Iic 3 := by sorry

-- Theorem for part (2)
theorem nonempty_intersection (a : ℝ) : (A ∩ B a).Nonempty → a ∈ Set.Ioi (5/2) := by sorry

end subset_condition_nonempty_intersection_l3551_355118


namespace double_after_increase_decrease_l3551_355148

theorem double_after_increase_decrease (r s N : ℝ) 
  (hr : r > 0) (hs : s > 0) (hN : N > 0) (hs_bound : s < 50) :
  N * (1 + r / 100) * (1 - s / 100) = 2 * N ↔ 
  r = (10000 + 100 * s) / (100 - s) :=
by sorry

end double_after_increase_decrease_l3551_355148


namespace equation_solution_l3551_355195

theorem equation_solution : ∃ x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) ↔ x = -10 := by
  sorry

end equation_solution_l3551_355195


namespace average_k_for_quadratic_roots_l3551_355190

theorem average_k_for_quadratic_roots (k : ℤ) : 
  let factors := [(1, 24), (2, 12), (3, 8), (4, 6)]
  let k_values := factors.map (λ (a, b) => a + b)
  let distinct_k_values := k_values.eraseDups
  (distinct_k_values.sum / distinct_k_values.length : ℚ) = 15 := by
  sorry

end average_k_for_quadratic_roots_l3551_355190


namespace greeting_card_distribution_four_l3551_355103

def greeting_card_distribution (n : ℕ) : ℕ :=
  if n = 4 then 9 else 0

theorem greeting_card_distribution_four :
  greeting_card_distribution 4 = 9 := by
  sorry

end greeting_card_distribution_four_l3551_355103


namespace multiple_with_four_digits_l3551_355132

theorem multiple_with_four_digits (k : ℕ) (h : k > 1) :
  ∃ w : ℕ, w > 0 ∧ k ∣ w ∧ w < k^4 ∧ 
  ∃ (a b c d : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (a = 0 ∨ a = 1 ∨ a = 8 ∨ a = 9) ∧
    (b = 0 ∨ b = 1 ∨ b = 8 ∨ b = 9) ∧
    (c = 0 ∨ c = 1 ∨ c = 8 ∨ c = 9) ∧
    (d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 9) ∧
    w = a * 1000 + b * 100 + c * 10 + d := by
  sorry

end multiple_with_four_digits_l3551_355132


namespace p_necessary_not_sufficient_for_q_l3551_355144

/-- The proposition p -/
def p (m a : ℝ) : Prop := m^2 - 4*a*m + 3*a^2 < 0 ∧ a < 0

/-- The proposition q -/
def q (m : ℝ) : Prop := ∀ x > 0, x + 4/x ≥ 1 - m

theorem p_necessary_not_sufficient_for_q :
  (∃ a m : ℝ, q m → p m a) ∧
  (∃ a m : ℝ, p m a ∧ ¬(q m)) ∧
  (∀ a : ℝ, (∃ m : ℝ, p m a ∧ q m) ↔ a ∈ Set.Icc (-1) 0) :=
sorry

end p_necessary_not_sufficient_for_q_l3551_355144


namespace stratified_sampling_second_year_selection_l3551_355134

theorem stratified_sampling_second_year_selection
  (total_students : ℕ)
  (first_year_students : ℕ)
  (second_year_students : ℕ)
  (first_year_selected : ℕ)
  (h1 : total_students = 70)
  (h2 : first_year_students = 30)
  (h3 : second_year_students = 40)
  (h4 : first_year_selected = 6)
  (h5 : total_students = first_year_students + second_year_students) :
  (first_year_selected : ℚ) / first_year_students * second_year_students = 8 := by
  sorry

end stratified_sampling_second_year_selection_l3551_355134


namespace range_of_m_value_of_m_l3551_355142

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + m - 2

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0

-- Define the additional condition
def additional_condition (x₁ x₂ : ℝ) : Prop :=
  (x₁ + 2) * (x₂ + 2) - 2 * x₁ * x₂ = 17

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  has_real_roots m → (m ≥ 2/3 ∧ m ≠ 1) :=
sorry

-- Theorem for the value of m given the additional condition
theorem value_of_m (m : ℝ) (x₁ x₂ : ℝ) :
  has_real_roots m →
  quadratic_equation m x₁ = 0 →
  quadratic_equation m x₂ = 0 →
  additional_condition x₁ x₂ →
  m = 3/2 :=
sorry

end range_of_m_value_of_m_l3551_355142


namespace midpoint_x_sum_l3551_355167

/-- Given a triangle in the Cartesian plane where the sum of x-coordinates of its vertices is 15,
    the sum of x-coordinates of the midpoints of its sides is also 15. -/
theorem midpoint_x_sum (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 := by
  sorry

end midpoint_x_sum_l3551_355167


namespace modulus_of_complex_fraction_l3551_355170

def i : ℂ := Complex.I

theorem modulus_of_complex_fraction : 
  Complex.abs ((1 + 3 * i) / (1 - 2 * i)) = Real.sqrt 2 := by
  sorry

end modulus_of_complex_fraction_l3551_355170


namespace root_sum_relation_l3551_355115

/-- The polynomial x^3 - 4x^2 + 7x - 10 -/
def p (x : ℝ) : ℝ := x^3 - 4*x^2 + 7*x - 10

/-- The sum of the k-th powers of the roots of p -/
def t (k : ℕ) : ℝ := sorry

theorem root_sum_relation :
  ∃ (u v w : ℝ), p u = 0 ∧ p v = 0 ∧ p w = 0 ∧
  (∀ k, t k = u^k + v^k + w^k) ∧
  t 0 = 3 ∧ t 1 = 4 ∧ t 2 = 10 ∧
  (∃ (d e f : ℝ), ∀ k ≥ 2, t (k+1) = d * t k + e * t (k-1) + f * t (k-2)) →
  ∃ (d e f : ℝ), (∀ k ≥ 2, t (k+1) = d * t k + e * t (k-1) + f * t (k-2)) ∧ d + e + f = 3 :=
by sorry

end root_sum_relation_l3551_355115


namespace polynomial_expansion_l3551_355193

theorem polynomial_expansion (x : ℝ) : 
  (3*x^3 + 4*x^2 + 12)*(x + 1) - (x + 1)*(2*x^3 + 6*x^2 - 42) + (6*x^2 - 28)*(x + 1)*(x - 2) = 
  7*x^4 - 7*x^3 - 42*x^2 + 82*x + 110 := by
sorry

end polynomial_expansion_l3551_355193


namespace volume_to_surface_area_ratio_l3551_355179

/-- A shape made of unit cubes -/
structure CubeShape where
  /-- The number of cubes in the base -/
  base_cubes : ℕ
  /-- The number of layers -/
  layers : ℕ
  /-- The total number of cubes -/
  total_cubes : ℕ
  /-- Condition: The base is a square -/
  base_is_square : base_cubes = 4
  /-- Condition: There are two layers -/
  two_layers : layers = 2
  /-- Condition: Total cubes is the product of base cubes and layers -/
  total_cubes_eq : total_cubes = base_cubes * layers

/-- The volume of the shape in cubic units -/
def volume (shape : CubeShape) : ℕ := shape.total_cubes

/-- The surface area of the shape in square units -/
def surface_area (shape : CubeShape) : ℕ :=
  6 * shape.total_cubes - 2 * shape.base_cubes

/-- The theorem stating the ratio of volume to surface area -/
theorem volume_to_surface_area_ratio (shape : CubeShape) :
  2 * (volume shape) = surface_area shape := by
  sorry

#check volume_to_surface_area_ratio

end volume_to_surface_area_ratio_l3551_355179


namespace gcd_cube_plus_27_l3551_355101

theorem gcd_cube_plus_27 (n : ℕ) (h : n > 27) :
  Nat.gcd (n^3 + 3^3) (n + 3) = n + 3 := by
  sorry

end gcd_cube_plus_27_l3551_355101


namespace circle_C_and_line_l_properties_l3551_355180

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, 2)

-- Define the line y = x
def line_y_eq_x (x : ℝ) : ℝ := x

-- Define the line l: y = kx + 1
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the line m: y = -2x + 4
def line_m (x : ℝ) : ℝ := -2 * x + 4

-- Define circle P
def circle_P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 5 * p.1^2 + 5 * p.2^2 - 16 * p.1 - 8 * p.2 + 12 = 0}

theorem circle_C_and_line_l_properties :
  ∃ (center : ℝ × ℝ) (P Q : ℝ × ℝ) (k : ℝ),
    center.2 = line_y_eq_x center.1 ∧
    point_A ∈ circle_C ∧
    point_B ∈ circle_C ∧
    P ∈ circle_C ∧
    Q ∈ circle_C ∧
    P.2 = line_l k P.1 ∧
    Q.2 = line_l k Q.1 ∧
    dot_product P Q = -2 →
    (∀ (p : ℝ × ℝ), p ∈ circle_C ↔ p.1^2 + p.2^2 = 4) ∧
    k = 0 ∧
    ∃ (E F : ℝ × ℝ),
      E ∈ circle_C ∧
      F ∈ circle_C ∧
      E.2 = line_m E.1 ∧
      F.2 = line_m F.1 ∧
      (2, 0) ∈ circle_P :=
by sorry

end circle_C_and_line_l_properties_l3551_355180


namespace rectangular_prism_volume_l3551_355106

/-- The volume of a rectangular prism with given edge lengths and space diagonal --/
theorem rectangular_prism_volume (AB AD AC1 : ℝ) :
  AB = 2 →
  AD = 2 →
  AC1 = 3 →
  ∃ (AA1 : ℝ), AA1 > 0 ∧ AB * AD * AA1 = 4 ∧ AC1^2 = AB^2 + AD^2 + AA1^2 :=
by sorry

end rectangular_prism_volume_l3551_355106


namespace projection_problem_l3551_355136

/-- Given two vectors that project onto the same vector, prove the resulting projection vector --/
theorem projection_problem (a b v p : ℝ × ℝ) : 
  a = (-3, 2) →
  b = (4, 5) →
  (∃ (k₁ k₂ : ℝ), p = k₁ • v ∧ p = k₂ • v) →
  p = (-69/58, 161/58) := by
  sorry

end projection_problem_l3551_355136


namespace triangle_solutions_l3551_355160

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    this theorem proves that if a = 6, b = 6√3, and A = π/6,
    then there are two possible solutions for the triangle. -/
theorem triangle_solutions (a b c : ℝ) (A B C : ℝ) :
  a = 6 →
  b = 6 * Real.sqrt 3 →
  A = π / 6 →
  (B = π / 3 ∧ C = π / 2 ∧ c = 12) ∨
  (B = 2 * π / 3 ∧ C = π / 6 ∧ c = 6) :=
sorry

end triangle_solutions_l3551_355160


namespace i_cubed_eq_neg_i_l3551_355155

-- Define the imaginary unit i
axiom i : ℂ
axiom i_squared : i^2 = -1

-- State the theorem
theorem i_cubed_eq_neg_i : i^3 = -i := by sorry

end i_cubed_eq_neg_i_l3551_355155


namespace rounds_played_l3551_355150

def total_points : ℕ := 154
def points_per_round : ℕ := 11

theorem rounds_played (total : ℕ) (per_round : ℕ) (h1 : total = total_points) (h2 : per_round = points_per_round) :
  total / per_round = 14 := by
  sorry

end rounds_played_l3551_355150


namespace green_peaches_count_l3551_355186

/-- Represents a basket of peaches -/
structure Basket where
  total : Nat
  red : Nat
  green : Nat

/-- The number of green peaches in a basket -/
def greenPeaches (b : Basket) : Nat := b.green

/-- Theorem: Given a basket with 10 total peaches and 7 red peaches, 
    the number of green peaches is 3 -/
theorem green_peaches_count (b : Basket) 
  (h1 : b.total = 10) 
  (h2 : b.red = 7) 
  (h3 : b.green = b.total - b.red) : 
  greenPeaches b = 3 := by
  sorry

#check green_peaches_count

end green_peaches_count_l3551_355186


namespace roots_sum_of_squares_l3551_355175

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 5*a + 5 = 0) → (b^2 - 5*b + 5 = 0) → a^2 + b^2 = 15 := by
  sorry

end roots_sum_of_squares_l3551_355175


namespace ed_lost_marbles_l3551_355161

theorem ed_lost_marbles (ed_initial : ℕ → ℕ) (doug_initial : ℕ) 
  (h1 : ed_initial doug_initial = doug_initial + 30)
  (h2 : ed_initial doug_initial - 21 = 91)
  (h3 : 91 = doug_initial + 9) : 
  ed_initial doug_initial - 91 = 21 :=
by sorry

end ed_lost_marbles_l3551_355161


namespace exp_13pi_over_3_rectangular_form_l3551_355126

open Complex

theorem exp_13pi_over_3_rectangular_form :
  exp (13 * π * I / 3) = (1 / 2 : ℂ) + (I * (Real.sqrt 3 / 2)) := by
  sorry

end exp_13pi_over_3_rectangular_form_l3551_355126


namespace sqrt_equation_solution_l3551_355100

theorem sqrt_equation_solution :
  ∀ x : ℝ, x > 0 → (6 * Real.sqrt (4 + x) + 6 * Real.sqrt (4 - x) = 9 * Real.sqrt 2) → x = Real.sqrt 255 / 4 := by
  sorry

end sqrt_equation_solution_l3551_355100


namespace triangle_circumradius_l3551_355159

/-- The circumradius of a triangle with sides 12, 10, and 7 is 6 units. -/
theorem triangle_circumradius (a b c : ℝ) (h_a : a = 12) (h_b : b = 10) (h_c : c = 7) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * area)
  R = 6 := by sorry

end triangle_circumradius_l3551_355159


namespace annette_sara_weight_difference_l3551_355151

/-- Given the weights of combinations of people, prove that Annette weighs 8 pounds more than Sara. -/
theorem annette_sara_weight_difference 
  (annette caitlin sara bob : ℝ) 
  (h1 : annette + caitlin = 95)
  (h2 : caitlin + sara = 87)
  (h3 : annette + sara = 97)
  (h4 : caitlin + bob = 100)
  (h5 : annette + caitlin + bob = 155) :
  annette - sara = 8 := by
sorry

end annette_sara_weight_difference_l3551_355151


namespace election_result_theorem_l3551_355114

/-- Represents the result of an election with five candidates -/
structure ElectionResult where
  total_votes : ℕ
  candidate1_votes : ℕ
  candidate2_votes : ℕ
  candidate3_votes : ℕ
  candidate4_votes : ℕ
  candidate5_votes : ℕ

/-- Theorem stating the election result given the conditions -/
theorem election_result_theorem (er : ElectionResult) : 
  er.candidate1_votes = (30 * er.total_votes) / 100 ∧
  er.candidate2_votes = (20 * er.total_votes) / 100 ∧
  er.candidate3_votes = 3000 ∧
  er.candidate3_votes = (15 * er.total_votes) / 100 ∧
  er.candidate4_votes = (25 * er.total_votes) / 100 ∧
  er.candidate5_votes = 2 * er.candidate3_votes →
  er.total_votes = 20000 ∧
  er.candidate1_votes = 6000 ∧
  er.candidate2_votes = 4000 ∧
  er.candidate3_votes = 3000 ∧
  er.candidate4_votes = 5000 ∧
  er.candidate5_votes = 6000 :=
by
  sorry

end election_result_theorem_l3551_355114


namespace final_rope_length_l3551_355182

def rope_lengths : List ℝ := [8, 20, 2, 2, 2, 7]
def knot_loss : ℝ := 1.2
def num_knots : ℕ := rope_lengths.length - 1

theorem final_rope_length :
  (rope_lengths.sum - num_knots * knot_loss : ℝ) = 35 := by
  sorry

end final_rope_length_l3551_355182
