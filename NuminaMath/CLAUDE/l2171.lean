import Mathlib

namespace line_intersection_x_axis_l2171_217144

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-coordinate of the intersection point of a line with the x-axis -/
def x_axis_intersection (l : Line) : ℝ :=
  sorry

theorem line_intersection_x_axis (l : Line) : 
  l.x₁ = 6 ∧ l.y₁ = 22 ∧ l.x₂ = -3 ∧ l.y₂ = 1 → x_axis_intersection l = -24/7 := by
  sorry

end line_intersection_x_axis_l2171_217144


namespace millicent_book_fraction_l2171_217152

/-- Given that:
    - Harold has 1/2 as many books as Millicent
    - Harold brings 1/3 of his books to the new home
    - The new home's library capacity is 5/6 of Millicent's old library capacity
    Prove that Millicent brings 2/3 of her books to the new home -/
theorem millicent_book_fraction (M : ℝ) (F : ℝ) (H : ℝ) :
  H = (1 / 2 : ℝ) * M →
  (1 / 3 : ℝ) * H + F * M = (5 / 6 : ℝ) * M →
  F = (2 / 3 : ℝ) := by
  sorry

end millicent_book_fraction_l2171_217152


namespace circle_center_radius_sum_l2171_217176

/-- Given a circle C with equation x^2 - 8y - 5 = -y^2 - 6x, 
    prove that a + b + r = 1 + √30, 
    where (a,b) is the center of C and r is its radius. -/
theorem circle_center_radius_sum (x y : ℝ) :
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 - 8*y - 5 = -y^2 - 6*x}
  ∃ (a b r : ℝ), (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
                 (a + b + r = 1 + Real.sqrt 30) :=
by sorry

end circle_center_radius_sum_l2171_217176


namespace like_terms_power_l2171_217132

/-- 
Given two monomials x^(a+3)y and -5xy^b that are like terms,
prove that (a+b)^2023 = -1
-/
theorem like_terms_power (a b : ℤ) : 
  (a + 3 = 1 ∧ b = 1) → (a + b)^2023 = -1 := by sorry

end like_terms_power_l2171_217132


namespace water_left_in_bathtub_water_left_is_7800_l2171_217139

/-- Calculates the amount of water left in a bathtub given specific conditions. -/
theorem water_left_in_bathtub 
  (faucet_drip_rate : ℝ) 
  (evaporation_rate : ℝ) 
  (time_running : ℝ) 
  (water_dumped : ℝ) : ℝ :=
  let water_added_per_hour := faucet_drip_rate * 60 - evaporation_rate
  let total_water_added := water_added_per_hour * time_running
  let water_remaining := total_water_added - water_dumped * 1000
  water_remaining

/-- Proves that under the given conditions, 7800 ml of water are left in the bathtub. -/
theorem water_left_is_7800 :
  water_left_in_bathtub 40 200 9 12 = 7800 := by
  sorry

end water_left_in_bathtub_water_left_is_7800_l2171_217139


namespace diamond_three_four_l2171_217185

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 36 := by
  sorry

end diamond_three_four_l2171_217185


namespace eleven_million_scientific_notation_l2171_217175

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem eleven_million_scientific_notation :
  toScientificNotation 11000000 = ScientificNotation.mk 1.1 7 (by norm_num) :=
sorry

end eleven_million_scientific_notation_l2171_217175


namespace second_quadrant_angle_sum_l2171_217109

theorem second_quadrant_angle_sum (θ : Real) : 
  (π / 2 < θ ∧ θ < π) →  -- θ is in the second quadrant
  (Real.tan (θ + π / 4) = 1 / 2) →  -- tan(θ + π/4) = 1/2
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end second_quadrant_angle_sum_l2171_217109


namespace parallelogram_area_triangle_area_l2171_217166

-- Define the parallelogram
def parallelogram_base : ℝ := 16
def parallelogram_height : ℝ := 25

-- Define the right-angled triangle
def triangle_side1 : ℝ := 3
def triangle_side2 : ℝ := 4

-- Theorem for parallelogram area
theorem parallelogram_area : 
  parallelogram_base * parallelogram_height = 400 := by sorry

-- Theorem for right-angled triangle area
theorem triangle_area : 
  (triangle_side1 * triangle_side2) / 2 = 6 := by sorry

end parallelogram_area_triangle_area_l2171_217166


namespace x_value_proof_l2171_217103

theorem x_value_proof (x : ℚ) 
  (eq1 : 8 * x^2 + 8 * x - 2 = 0) 
  (eq2 : 32 * x^2 + 68 * x - 8 = 0) : 
  x = 1/8 := by
sorry

end x_value_proof_l2171_217103


namespace range_of_a_l2171_217168

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) 
  (h2 : ∃ x : ℝ, x^2 + 4*x + a = 0) : 
  a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end range_of_a_l2171_217168


namespace smallest_b_value_l2171_217142

theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : a + b = 7)
  (h4 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2)) : 
  (∀ b' : ℝ, (∃ a' : ℝ, 2 < a' ∧ a' < b' ∧ a' + b' = 7 ∧
    ¬ (2 + a' > b' ∧ 2 + b' > a' ∧ a' + b' > 2)) → b' ≥ 9/2) ∧ b = 9/2 :=
by sorry

end smallest_b_value_l2171_217142


namespace alice_age_l2171_217187

theorem alice_age (alice_age mother_age : ℕ) 
  (h1 : alice_age = mother_age - 18)
  (h2 : alice_age + mother_age = 50) : 
  alice_age = 16 := by sorry

end alice_age_l2171_217187


namespace solution_existence_l2171_217126

theorem solution_existence (m : ℝ) : 
  (∃ x : ℝ, 3 * Real.sin x + 4 * Real.cos x = 2 * m - 1) ↔ 
  -2 ≤ m ∧ m ≤ 3 :=
by sorry

end solution_existence_l2171_217126


namespace jar_contents_l2171_217155

-- Define the number of candy pieces
def candy_pieces : Float := 3409.0

-- Define the number of secret eggs
def secret_eggs : Float := 145.0

-- Define the total number of items
def total_items : Float := candy_pieces + secret_eggs

-- Theorem statement
theorem jar_contents : total_items = 3554.0 := by
  sorry

end jar_contents_l2171_217155


namespace flash_fraction_is_one_l2171_217158

/-- The fraction of an hour it takes for a light to flash 120 times, given that it flashes every 30 seconds -/
def flash_fraction : ℚ :=
  let flash_interval : ℚ := 30 / 3600  -- 30 seconds as a fraction of an hour
  let total_flashes : ℕ := 120
  total_flashes * flash_interval

theorem flash_fraction_is_one : flash_fraction = 1 := by
  sorry

end flash_fraction_is_one_l2171_217158


namespace johnson_family_seating_l2171_217169

def num_sons : ℕ := 5
def num_daughters : ℕ := 4
def total_children : ℕ := num_sons + num_daughters

def total_arrangements : ℕ := Nat.factorial total_children

def arrangements_without_bbg : ℕ := Nat.factorial 7 * 4

theorem johnson_family_seating :
  total_arrangements - arrangements_without_bbg = 342720 := by
  sorry

end johnson_family_seating_l2171_217169


namespace joan_apples_l2171_217123

/-- The number of apples Joan has after picking, receiving, and having some taken away. -/
def final_apples (initial : ℕ) (added : ℕ) (taken : ℕ) : ℕ :=
  initial + added - taken

/-- Theorem stating that Joan's final number of apples is 55 given the problem conditions. -/
theorem joan_apples : final_apples 43 27 15 = 55 := by
  sorry

end joan_apples_l2171_217123


namespace new_person_weight_l2171_217182

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (average_increase : ℝ) : ℝ :=
  initial_count * average_increase + replaced_weight

/-- Theorem stating that the weight of the new person is 93 kg -/
theorem new_person_weight :
  weight_of_new_person 8 65 3.5 = 93 := by
  sorry

end new_person_weight_l2171_217182


namespace power_three_nineteen_mod_ten_l2171_217171

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by sorry

end power_three_nineteen_mod_ten_l2171_217171


namespace preimage_of_3_1_l2171_217129

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

/-- Theorem stating that (2, -1) is the pre-image of (3, 1) under f -/
theorem preimage_of_3_1 : f (2, -1) = (3, 1) := by
  sorry

end preimage_of_3_1_l2171_217129


namespace greatest_consecutive_mixed_number_l2171_217194

/-- 
Given 6 consecutive mixed numbers with a sum of 75.5, 
prove that the greatest number is 15 1/12.
-/
theorem greatest_consecutive_mixed_number :
  ∀ (a b c d e f : ℚ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f →  -- consecutive
    b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 →  -- mixed numbers
    a + b + c + d + e + f = 75.5 →  -- sum condition
    f = 15 + 1/12 :=  -- greatest number
by sorry

end greatest_consecutive_mixed_number_l2171_217194


namespace distribute_6_3_l2171_217192

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_6_3 : distribute 6 3 = 729 := by sorry

end distribute_6_3_l2171_217192


namespace car_b_speed_l2171_217157

/-- Proves that given the initial conditions and final state, Car B's speed is 50 mph -/
theorem car_b_speed (initial_distance : ℝ) (car_a_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 30 →
  car_a_speed = 58 →
  time = 4.75 →
  final_distance = 8 →
  (car_a_speed * time - initial_distance - final_distance) / time = 50 := by
sorry

end car_b_speed_l2171_217157


namespace polynomial_equality_l2171_217122

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 3) = x^2 + 2*x - b) → a - b = -10 := by
  sorry

end polynomial_equality_l2171_217122


namespace A_specific_value_l2171_217159

def A : ℕ → ℕ
  | 0 => 1
  | n + 1 => A (n / 2023) + A (n / 2023^2) + A (n / 2023^3)

theorem A_specific_value : A (2023^(3^2) + 20) = 653 := by
  sorry

end A_specific_value_l2171_217159


namespace ellipse_foci_y_axis_l2171_217112

theorem ellipse_foci_y_axis (k : ℝ) :
  (∀ x y : ℝ, x^2 / (9 - k) + y^2 / (k - 4) = 1) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ b^2 = a^2 + c^2 ∧
    ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / (9 - k) + y^2 / (k - 4) = 1) →
  13/2 < k ∧ k < 9 :=
by sorry

end ellipse_foci_y_axis_l2171_217112


namespace inequality_solution_set_l2171_217107

theorem inequality_solution_set :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1 ∨ x > 3} := by sorry

end inequality_solution_set_l2171_217107


namespace davids_homework_time_l2171_217124

theorem davids_homework_time (math_time spelling_time reading_time : ℕ) 
  (h1 : math_time = 15)
  (h2 : spelling_time = 18)
  (h3 : reading_time = 27) :
  math_time + spelling_time + reading_time = 60 := by
  sorry

end davids_homework_time_l2171_217124


namespace arithmetic_sequence_sum_l2171_217120

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 5 + a 8 + a 11 = 48 →
  a 6 + a 7 = 24 := by
  sorry

end arithmetic_sequence_sum_l2171_217120


namespace root_implies_m_value_l2171_217153

theorem root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x - 3 = 0 ∧ x = 3) → m = 2 := by
  sorry

end root_implies_m_value_l2171_217153


namespace f_2015_value_l2171_217146

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_shift (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x + f 2

theorem f_2015_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period_shift f)
  (h_f1 : f 1 = 2) :
  f 2015 = -2 := by
sorry

end f_2015_value_l2171_217146


namespace rectangle_triangle_area_equality_l2171_217150

/-- Given a square ABCD with side length 2 and a rectangle EFGH within it,
    prove that if AE = x, EB = 1, EF = x, and the areas of EFGH and ABE are equal,
    then x = 3/2 -/
theorem rectangle_triangle_area_equality (x : ℝ) : 
  (∀ (A B C D E F G H : ℝ × ℝ),
    -- ABCD is a square with side length 2
    ‖B - A‖ = 2 ∧ ‖C - B‖ = 2 ∧ ‖D - C‖ = 2 ∧ ‖A - D‖ = 2 ∧
    -- EFGH is a rectangle within the square
    (E.1 ≥ A.1 ∧ E.1 ≤ B.1) ∧ (E.2 ≥ A.2 ∧ E.2 ≤ D.2) ∧
    (F.1 ≥ A.1 ∧ F.1 ≤ B.1) ∧ (F.2 ≥ A.2 ∧ F.2 ≤ D.2) ∧
    (G.1 ≥ A.1 ∧ G.1 ≤ B.1) ∧ (G.2 ≥ A.2 ∧ G.2 ≤ D.2) ∧
    (H.1 ≥ A.1 ∧ H.1 ≤ B.1) ∧ (H.2 ≥ A.2 ∧ H.2 ≤ D.2) ∧
    -- AE = x, EB = 1, EF = x
    ‖E - A‖ = x ∧ ‖B - E‖ = 1 ∧ ‖F - E‖ = x ∧
    -- Areas of rectangle EFGH and triangle ABE are equal
    ‖F - E‖ * ‖G - F‖ = (1/2) * ‖E - A‖ * ‖B - E‖) →
  x = 3/2 := by
  sorry


end rectangle_triangle_area_equality_l2171_217150


namespace intersection_difference_l2171_217149

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1
def parabola2 (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

-- Define the theorem
theorem intersection_difference (a b c d : ℝ) 
  (h1 : parabola1 a = parabola2 a) 
  (h2 : parabola1 c = parabola2 c) 
  (h3 : c ≥ a) : 
  c - a = 6/5 := by
  sorry

end intersection_difference_l2171_217149


namespace lcm_of_20_45_75_l2171_217117

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end lcm_of_20_45_75_l2171_217117


namespace ellipse_eccentricity_l2171_217136

/-- Represents an ellipse with semi-major axis 'a' and semi-minor axis 2 -/
structure Ellipse where
  a : ℝ
  h_a : a > 2

/-- Represents a line with equation y = x - 2 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 2}

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- A focus of the ellipse -/
def focus (e : Ellipse) : ℝ × ℝ :=
  sorry

theorem ellipse_eccentricity (e : Ellipse) :
  focus e ∈ Line →
  eccentricity e = Real.sqrt 2 / 2 :=
sorry

end ellipse_eccentricity_l2171_217136


namespace factoring_left_to_right_l2171_217110

theorem factoring_left_to_right (m n : ℝ) : m^2 - 2*m*n + n^2 = (m - n)^2 := by
  sorry

end factoring_left_to_right_l2171_217110


namespace polynomial_value_l2171_217151

/-- Given a polynomial G satisfying certain conditions, prove G(8) = 491/3 -/
theorem polynomial_value (G : ℝ → ℝ) : 
  (∀ x, G (4 * x) / G (x + 2) = 4 - (20 * x + 24) / (x^2 + 4 * x + 4)) →
  G 4 = 35 →
  G 8 = 491 / 3 := by
sorry

end polynomial_value_l2171_217151


namespace c_prime_coordinates_l2171_217108

/-- Triangle ABC with vertices A(1,2), B(2,1), and C(3,2) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Similar triangle A'B'C' with similarity ratio 2 and origin as center of similarity -/
def similarTriangle (t : Triangle) : Triangle :=
  { A := (2 * t.A.1, 2 * t.A.2),
    B := (2 * t.B.1, 2 * t.B.2),
    C := (2 * t.C.1, 2 * t.C.2) }

/-- The original triangle ABC -/
def ABC : Triangle :=
  { A := (1, 2),
    B := (2, 1),
    C := (3, 2) }

/-- Theorem stating that C' has coordinates (6,4) or (-6,-4) -/
theorem c_prime_coordinates :
  let t' := similarTriangle ABC
  (t'.C = (6, 4) ∨ t'.C = (-6, -4)) :=
sorry

end c_prime_coordinates_l2171_217108


namespace chemical_reaction_result_l2171_217130

-- Define the initial amounts
def initial_silver_nitrate : ℝ := 2
def initial_sodium_hydroxide : ℝ := 2
def initial_hydrochloric_acid : ℝ := 0.5

-- Define the reactions
def main_reaction (x : ℝ) : ℝ := x
def side_reaction (x : ℝ) : ℝ := x

-- Theorem statement
theorem chemical_reaction_result :
  let sodium_hydroxide_in_side_reaction := min initial_sodium_hydroxide initial_hydrochloric_acid
  let remaining_sodium_hydroxide := initial_sodium_hydroxide - sodium_hydroxide_in_side_reaction
  let reaction_limit := min remaining_sodium_hydroxide initial_silver_nitrate
  let sodium_nitrate_formed := main_reaction reaction_limit
  let silver_chloride_formed := main_reaction reaction_limit
  let unreacted_sodium_hydroxide := remaining_sodium_hydroxide - reaction_limit
  sodium_nitrate_formed = 1.5 ∧ 
  silver_chloride_formed = 1.5 ∧ 
  unreacted_sodium_hydroxide = 0 :=
by
  sorry


end chemical_reaction_result_l2171_217130


namespace largest_package_size_l2171_217125

theorem largest_package_size (juan_markers alicia_markers : ℕ) 
  (h1 : juan_markers = 36) (h2 : alicia_markers = 48) : 
  Nat.gcd juan_markers alicia_markers = 12 := by
  sorry

end largest_package_size_l2171_217125


namespace distance_AB_on_parabola_l2171_217135

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem distance_AB_on_parabola (A : ℝ × ℝ) :
  parabola A.1 A.2 →  -- A lies on the parabola
  ‖A - focus‖ = ‖point_B - focus‖ →  -- |AF| = |BF|
  ‖A - point_B‖ = 2 * Real.sqrt 2 :=  -- |AB| = 2√2
by sorry

end distance_AB_on_parabola_l2171_217135


namespace sergey_mistake_l2171_217104

theorem sergey_mistake : ¬∃ a : ℤ, a % 15 = 8 ∧ a % 20 = 17 := by
  sorry

end sergey_mistake_l2171_217104


namespace sqrt_equation_equivalence_l2171_217172

theorem sqrt_equation_equivalence (x : ℝ) (h : x > 9) :
  (Real.sqrt (x - 9 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 9 * Real.sqrt (x - 9)) - 3) ↔ 
  x ≥ 40.5 := by
sorry

end sqrt_equation_equivalence_l2171_217172


namespace greatest_divisor_with_remainder_l2171_217131

def a : ℕ := 3547
def b : ℕ := 12739
def c : ℕ := 21329
def r : ℕ := 17

theorem greatest_divisor_with_remainder (d : ℕ) : d > 0 → d.gcd (a - r) = d → d.gcd (b - r) = d → d.gcd (c - r) = d → 
  (∀ k : ℕ, k > d → (k.gcd (a - r) ≠ k ∨ k.gcd (b - r) ≠ k ∨ k.gcd (c - r) ≠ k)) → 
  (∀ n : ℕ, n > 0 → (a % n = r ∧ b % n = r ∧ c % n = r) → n ≤ d) :=
by sorry

end greatest_divisor_with_remainder_l2171_217131


namespace framed_painting_ratio_l2171_217163

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framedDimensions (fp : FramedPainting) : ℝ × ℝ :=
  (fp.painting_width + 2 * fp.side_frame_width, fp.painting_height + 4 * fp.side_frame_width)

/-- Calculates the area of the frame -/
def frameArea (fp : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions fp
  w * h - fp.painting_width * fp.painting_height

/-- Theorem stating the ratio of dimensions for a specific framed painting -/
theorem framed_painting_ratio :
  ∃ (fp : FramedPainting),
    fp.painting_width = 15 ∧
    fp.painting_height = 30 ∧
    frameArea fp = fp.painting_width * fp.painting_height ∧
    let (w, h) := framedDimensions fp
    w / h = 1 / 2 := by
  sorry

end framed_painting_ratio_l2171_217163


namespace integral_proof_l2171_217105

open Real

theorem integral_proof (x : ℝ) (h : x ≠ -2 ∧ x ≠ 8) : 
  deriv (fun x => (1/10) * log (abs ((x - 8) / (x + 2)))) x = 1 / (x^2 - 6*x - 16) :=
sorry

end integral_proof_l2171_217105


namespace complex_real_condition_l2171_217160

/-- If z = m^2 - 1 + (m-1)i is a real number and m is real, then m = 1 -/
theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := m^2 - 1 + (m - 1) * Complex.I
  (z.im = 0) → m = 1 := by
sorry

end complex_real_condition_l2171_217160


namespace room_painting_cost_l2171_217148

/-- Calculate the cost of painting a room's walls given its dimensions and openings. -/
def paintingCost (roomLength roomWidth roomHeight : ℝ)
                 (doorCount doorLength doorHeight : ℝ)
                 (largeWindowCount largeWindowLength largeWindowHeight : ℝ)
                 (smallWindowCount smallWindowLength smallWindowHeight : ℝ)
                 (costPerSqm : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := doorCount * doorLength * doorHeight
  let largeWindowArea := largeWindowCount * largeWindowLength * largeWindowHeight
  let smallWindowArea := smallWindowCount * smallWindowLength * smallWindowHeight
  let paintableArea := wallArea - (doorArea + largeWindowArea + smallWindowArea)
  paintableArea * costPerSqm

/-- Theorem stating that the cost of painting the room with given dimensions is 474 Rs. -/
theorem room_painting_cost :
  paintingCost 10 7 5 2 1 3 1 2 1.5 2 1 1.5 3 = 474 := by
  sorry

end room_painting_cost_l2171_217148


namespace negation_of_proposition_l2171_217154

theorem negation_of_proposition (a b : ℝ) :
  ¬(a + b > 0 → a > 0 ∧ b > 0) ↔ (a + b ≤ 0 → a ≤ 0 ∨ b ≤ 0) := by
  sorry

end negation_of_proposition_l2171_217154


namespace min_apples_in_basket_sixty_two_satisfies_conditions_min_apples_is_sixty_two_l2171_217173

theorem min_apples_in_basket (N : ℕ) : 
  (N % 3 = 2) ∧ (N % 4 = 2) ∧ (N % 5 = 2) → N ≥ 62 :=
by sorry

theorem sixty_two_satisfies_conditions : 
  (62 % 3 = 2) ∧ (62 % 4 = 2) ∧ (62 % 5 = 2) :=
by sorry

theorem min_apples_is_sixty_two : 
  ∃ (N : ℕ), (N % 3 = 2) ∧ (N % 4 = 2) ∧ (N % 5 = 2) ∧ N = 62 :=
by sorry

end min_apples_in_basket_sixty_two_satisfies_conditions_min_apples_is_sixty_two_l2171_217173


namespace max_students_above_average_l2171_217102

theorem max_students_above_average (n : ℕ) (score1 score2 : ℚ) : 
  n = 150 →
  score1 > score2 →
  (n - 1) * score1 + score2 > n * ((n - 1) * score1 + score2) / n →
  ∃ (m : ℕ), m ≤ n ∧ m = 149 ∧ 
    (∀ (k : ℕ), k > m → 
      k * score1 + (n - k) * score2 ≤ n * (k * score1 + (n - k) * score2) / n) :=
by sorry

end max_students_above_average_l2171_217102


namespace fraction_to_zero_power_l2171_217161

theorem fraction_to_zero_power :
  let x : ℚ := -123456789 / 9876543210
  x ≠ 0 →
  x^0 = 1 := by sorry

end fraction_to_zero_power_l2171_217161


namespace min_value_of_z_l2171_217178

theorem min_value_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) : 
  ∀ z : ℝ, z = x^2 + 4*y^2 → z ≥ 4 :=
by sorry

end min_value_of_z_l2171_217178


namespace corporation_employees_l2171_217121

theorem corporation_employees (total : ℕ) (part_time : ℕ) (full_time : ℕ) :
  total = 65134 →
  part_time = 2041 →
  full_time = total - part_time →
  full_time = 63093 := by
sorry

end corporation_employees_l2171_217121


namespace jordans_mangoes_l2171_217137

theorem jordans_mangoes (total : ℕ) (ripe : ℕ) (unripe : ℕ) (kept : ℕ) (jars : ℕ) (mangoes_per_jar : ℕ) : 
  ripe = total / 3 →
  unripe = 2 * total / 3 →
  kept = 16 →
  jars = 5 →
  mangoes_per_jar = 4 →
  unripe = kept + jars * mangoes_per_jar →
  total = ripe + unripe →
  total = 54 :=
by sorry

end jordans_mangoes_l2171_217137


namespace complement_of_A_in_U_l2171_217165

universe u

def U : Set ℕ := {2, 4, 6, 8, 9}
def A : Set ℕ := {2, 4, 9}

theorem complement_of_A_in_U :
  (U \ A) = {6, 8} := by sorry

end complement_of_A_in_U_l2171_217165


namespace function_satisfying_property_is_square_l2171_217184

open Real

-- Define the property for the function
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = ⨆ y : ℝ, (2 * x * y - f y)

-- Theorem statement
theorem function_satisfying_property_is_square (f : ℝ → ℝ) :
  SatisfiesProperty f → ∀ x : ℝ, f x = x^2 := by
  sorry


end function_satisfying_property_is_square_l2171_217184


namespace collins_savings_l2171_217167

-- Define the constants
def cans_at_home : ℕ := 12
def cans_from_neighbor : ℕ := 46
def cans_from_office : ℕ := 250
def price_per_can : ℚ := 25 / 100

-- Define the functions
def cans_at_grandparents : ℕ := 3 * cans_at_home

def total_cans : ℕ := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_office

def total_money : ℚ := (total_cans : ℚ) * price_per_can

def savings_amount : ℚ := total_money / 2

-- Theorem statement
theorem collins_savings : savings_amount = 43 := by
  sorry

end collins_savings_l2171_217167


namespace merry_go_round_revolutions_l2171_217119

/-- The number of revolutions needed for two horses on a merry-go-round to travel the same distance -/
theorem merry_go_round_revolutions (r₁ r₂ n₁ : ℝ) (hr₁ : r₁ = 15) (hr₂ : r₂ = 5) (hn₁ : n₁ = 20) :
  ∃ n₂ : ℝ, n₂ = 60 ∧ n₁ * r₁ = n₂ * r₂ := by
  sorry


end merry_go_round_revolutions_l2171_217119


namespace sin_sum_from_sin_cos_sums_l2171_217133

theorem sin_sum_from_sin_cos_sums (x y : Real) 
  (h1 : Real.sin x + Real.sin y = Real.sqrt 2 / 2)
  (h2 : Real.cos x + Real.cos y = Real.sqrt 6 / 2) :
  Real.sin (x + y) = Real.sqrt 3 / 2 := by
sorry

end sin_sum_from_sin_cos_sums_l2171_217133


namespace distance_incenter_circumcenter_squared_l2171_217170

-- Define a 30-60-90 right triangle with hypotenuse 2
structure Triangle30_60_90 where
  hypotenuse : ℝ
  is_30_60_90 : hypotenuse = 2

-- Define the distance between incenter and circumcenter
def distance_incenter_circumcenter (t : Triangle30_60_90) : ℝ := sorry

theorem distance_incenter_circumcenter_squared (t : Triangle30_60_90) :
  (distance_incenter_circumcenter t)^2 = 2 - Real.sqrt 3 := by sorry

end distance_incenter_circumcenter_squared_l2171_217170


namespace total_rackets_packed_l2171_217134

/-- Proves that given the conditions of racket packaging, the total number of rackets is 100 -/
theorem total_rackets_packed (total_cartons : ℕ) (three_racket_cartons : ℕ) 
  (h1 : total_cartons = 38)
  (h2 : three_racket_cartons = 24) :
  3 * three_racket_cartons + 2 * (total_cartons - three_racket_cartons) = 100 := by
  sorry

end total_rackets_packed_l2171_217134


namespace julia_dimes_count_l2171_217195

theorem julia_dimes_count : ∃ d : ℕ, 
  20 < d ∧ d < 200 ∧ 
  d % 6 = 1 ∧ 
  d % 7 = 1 ∧ 
  d % 8 = 1 ∧ 
  d = 169 := by sorry

end julia_dimes_count_l2171_217195


namespace b_range_for_inequality_l2171_217186

/-- Given an inequality ax + b > 2(x + 1) with solution set {x | x < 1}, 
    prove that the range of values for b is (4, +∞) -/
theorem b_range_for_inequality (a b : ℝ) : 
  (∀ x, ax + b > 2*(x + 1) ↔ x < 1) → 
  ∃ y, y > 4 ∧ b > y :=
sorry

end b_range_for_inequality_l2171_217186


namespace total_pencils_l2171_217101

theorem total_pencils (num_people : ℕ) (pencils_per_person : ℕ) : 
  num_people = 5 → pencils_per_person = 15 → num_people * pencils_per_person = 75 := by
  sorry

end total_pencils_l2171_217101


namespace range_of_fraction_l2171_217174

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  (∃ (x₁ y₁ : ℝ), 1 ≤ x₁ ∧ x₁ ≤ 4 ∧ 3 ≤ y₁ ∧ y₁ ≤ 6 ∧ x₁ / y₁ = 1/6) ∧
  (∃ (x₂ y₂ : ℝ), 1 ≤ x₂ ∧ x₂ ≤ 4 ∧ 3 ≤ y₂ ∧ y₂ ≤ 6 ∧ x₂ / y₂ = 4/3) ∧
  (∀ (x' y' : ℝ), 1 ≤ x' ∧ x' ≤ 4 → 3 ≤ y' ∧ y' ≤ 6 → 1/6 ≤ x' / y' ∧ x' / y' ≤ 4/3) :=
by sorry

end range_of_fraction_l2171_217174


namespace proposition_and_converse_l2171_217181

theorem proposition_and_converse (a b : ℝ) : 
  (((a + b ≥ 2) → (a ≥ 1 ∨ b ≥ 1)) ∧ 
  (∃ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) ∧ ¬(a + b ≥ 2))) :=
by sorry

end proposition_and_converse_l2171_217181


namespace geometric_sequence_problem_l2171_217128

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  q > 1 →  -- common ratio > 1
  4 * (a 2005)^2 - 8 * (a 2005) + 3 = 0 →  -- a₂₀₀₅ is a root
  4 * (a 2006)^2 - 8 * (a 2006) + 3 = 0 →  -- a₂₀₀₆ is a root
  a 2007 + a 2008 = 18 := by
sorry

end geometric_sequence_problem_l2171_217128


namespace simplify_fraction_l2171_217143

theorem simplify_fraction : 
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end simplify_fraction_l2171_217143


namespace combinations_theorem_l2171_217106

/-- The number of choices in the art group -/
def art_choices : ℕ := 2

/-- The number of choices in the sports group -/
def sports_choices : ℕ := 3

/-- The number of choices in the music group -/
def music_choices : ℕ := 4

/-- The total number of possible combinations -/
def total_combinations : ℕ := art_choices * sports_choices * music_choices

theorem combinations_theorem : total_combinations = 24 := by
  sorry

end combinations_theorem_l2171_217106


namespace diesel_consumption_calculation_l2171_217196

/-- Calculates the diesel consumption of a car given its fuel efficiency, travel time, and speed. -/
theorem diesel_consumption_calculation
  (fuel_efficiency : ℝ)  -- Diesel consumption in liters per kilometer
  (travel_time : ℝ)      -- Travel time in hours
  (speed : ℝ)            -- Speed in kilometers per hour
  (h1 : fuel_efficiency = 0.14)
  (h2 : travel_time = 2.5)
  (h3 : speed = 93.6) :
  fuel_efficiency * travel_time * speed = 32.76 := by
    sorry

#check diesel_consumption_calculation

end diesel_consumption_calculation_l2171_217196


namespace equal_group_formations_20_people_l2171_217199

/-- The number of ways to form a group with an equal number of boys and girls -/
def equalGroupFormations (totalPeople boys girls : ℕ) : ℕ :=
  Nat.choose totalPeople boys

/-- Theorem stating that the number of ways to form a group with an equal number
    of boys and girls from 20 people (10 boys and 10 girls) is equal to C(20,10) -/
theorem equal_group_formations_20_people :
  equalGroupFormations 20 10 10 = Nat.choose 20 10 := by
  sorry

#eval equalGroupFormations 20 10 10

end equal_group_formations_20_people_l2171_217199


namespace jacket_price_calculation_l2171_217111

def calculate_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (coupon : ℝ) (tax : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let price_after_coupon := price_after_discount2 - coupon
  price_after_coupon * (1 + tax)

theorem jacket_price_calculation :
  calculate_final_price 150 0.30 0.10 10 0.05 = 88.725 := by
  sorry

end jacket_price_calculation_l2171_217111


namespace chocolate_bunny_value_is_100_l2171_217193

/-- The number of points needed to win the Nintendo Switch -/
def total_points_needed : ℕ := 2000

/-- The number of chocolate bunnies already sold -/
def chocolate_bunnies_sold : ℕ := 8

/-- The number of points earned per Snickers bar -/
def points_per_snickers : ℕ := 25

/-- The number of Snickers bars needed to win the Nintendo Switch -/
def snickers_bars_needed : ℕ := 48

/-- The value of each chocolate bunny in points -/
def chocolate_bunny_value : ℕ := (total_points_needed - (points_per_snickers * snickers_bars_needed)) / chocolate_bunnies_sold

theorem chocolate_bunny_value_is_100 : chocolate_bunny_value = 100 := by
  sorry

end chocolate_bunny_value_is_100_l2171_217193


namespace complex_number_in_second_quadrant_l2171_217177

/-- The complex number z = i(3+i) corresponds to a point in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : ∃ (x y : ℝ), Complex.I * (3 + Complex.I) = Complex.mk x y ∧ x < 0 ∧ y > 0 := by
  sorry

end complex_number_in_second_quadrant_l2171_217177


namespace heart_ratio_equals_one_l2171_217156

def heart (n m : ℕ) : ℕ := n^3 * m^3

theorem heart_ratio_equals_one : (heart 3 5) / (heart 5 3) = 1 := by
  sorry

end heart_ratio_equals_one_l2171_217156


namespace equal_cost_at_280_minutes_unique_equal_cost_point_l2171_217145

/-- Represents a phone service plan with a monthly fee and per-minute rate. -/
structure ServicePlan where
  monthlyFee : ℝ
  perMinuteRate : ℝ

/-- Calculates the cost of a service plan for a given number of minutes. -/
def planCost (plan : ServicePlan) (minutes : ℝ) : ℝ :=
  plan.monthlyFee + plan.perMinuteRate * minutes

/-- Theorem stating that the costs of two specific phone service plans are equal at 280 minutes. -/
theorem equal_cost_at_280_minutes : 
  let plan1 : ServicePlan := { monthlyFee := 22, perMinuteRate := 0.13 }
  let plan2 : ServicePlan := { monthlyFee := 8, perMinuteRate := 0.18 }
  planCost plan1 280 = planCost plan2 280 := by
  sorry

/-- Theorem stating that 280 minutes is the unique point where the costs are equal. -/
theorem unique_equal_cost_point : 
  let plan1 : ServicePlan := { monthlyFee := 22, perMinuteRate := 0.13 }
  let plan2 : ServicePlan := { monthlyFee := 8, perMinuteRate := 0.18 }
  ∀ x : ℝ, planCost plan1 x = planCost plan2 x ↔ x = 280 := by
  sorry

end equal_cost_at_280_minutes_unique_equal_cost_point_l2171_217145


namespace not_divisible_by_121_l2171_217127

theorem not_divisible_by_121 (n : ℤ) : ¬(121 ∣ (n^2 + 3*n + 5)) := by
  sorry

end not_divisible_by_121_l2171_217127


namespace fraction_sum_difference_l2171_217180

theorem fraction_sum_difference (a b c d e f : ℤ) :
  (a : ℚ) / b + (c : ℚ) / d - (e : ℚ) / f = (53 : ℚ) / 72 :=
by
  -- The proof would go here
  sorry

end fraction_sum_difference_l2171_217180


namespace batsman_average_theorem_l2171_217114

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  lastInningScore : ℕ
  averageIncrease : ℚ

/-- Calculates the new average of a batsman after their latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.totalRuns + b.lastInningScore) / b.innings

theorem batsman_average_theorem (b : Batsman) 
  (h1 : b.innings = 17)
  (h2 : b.lastInningScore = 85)
  (h3 : b.averageIncrease = 3)
  (h4 : newAverage b = (b.totalRuns / (b.innings - 1) + b.averageIncrease)) :
  newAverage b = 37 := by
  sorry

end batsman_average_theorem_l2171_217114


namespace equation_solutions_l2171_217116

theorem equation_solutions : 
  (∃ S₁ : Set ℝ, S₁ = {x : ℝ | x * (x - 2) + x - 2 = 0} ∧ S₁ = {2, -1}) ∧
  (∃ S₂ : Set ℝ, S₂ = {x : ℝ | 2 * x^2 + 5 * x + 3 = 0} ∧ S₂ = {-1, -3/2}) :=
by
  sorry


end equation_solutions_l2171_217116


namespace smallest_divisor_of_7614_l2171_217140

def n : ℕ := 7614

theorem smallest_divisor_of_7614 :
  ∃ (d : ℕ), d > 1 ∧ d ∣ n ∧ ∀ (k : ℕ), 1 < k ∧ k ∣ n → d ≤ k :=
by sorry

end smallest_divisor_of_7614_l2171_217140


namespace fixed_points_equality_l2171_217162

def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def FixedPoints (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = x}

theorem fixed_points_equality
  (f : ℝ → ℝ)
  (h_inj : Function.Injective f)
  (h_incr : StrictlyIncreasing f) :
  FixedPoints f = FixedPoints (f ∘ f) := by
  sorry

end fixed_points_equality_l2171_217162


namespace second_smallest_perimeter_l2171_217197

/-- A triangle with consecutive integer side lengths -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 1 ∧ c = b + 1
  is_triangle : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ :=
  t.a + t.b + t.c

/-- The second smallest perimeter of a triangle with consecutive integer side lengths is 12 -/
theorem second_smallest_perimeter :
  ∃ (t : ConsecutiveIntegerTriangle), perimeter t = 12 ∧
  ∀ (s : ConsecutiveIntegerTriangle), perimeter s ≠ 9 → perimeter s ≥ 12 :=
by sorry

end second_smallest_perimeter_l2171_217197


namespace sacks_filled_twice_l2171_217179

/-- Represents the number of times sacks can be filled with wood --/
def times_sacks_filled (father_capacity : ℕ) (ranger_capacity : ℕ) (volunteer_capacity : ℕ) (num_volunteers : ℕ) (total_wood : ℕ) : ℕ :=
  total_wood / (father_capacity + ranger_capacity + num_volunteers * volunteer_capacity)

/-- Theorem stating that under the given conditions, the sacks can be filled 2 times --/
theorem sacks_filled_twice :
  times_sacks_filled 20 30 25 2 200 = 2 := by
  sorry

end sacks_filled_twice_l2171_217179


namespace greatest_x_under_conditions_l2171_217141

theorem greatest_x_under_conditions (x : ℕ) 
  (h1 : x > 0) 
  (h2 : ∃ k : ℕ, x = 5 * k) 
  (h3 : x^3 < 1331) : 
  ∀ y : ℕ, (y > 0 ∧ (∃ m : ℕ, y = 5 * m) ∧ y^3 < 1331) → y ≤ 10 :=
sorry

end greatest_x_under_conditions_l2171_217141


namespace delta_value_l2171_217115

theorem delta_value : ∃ Δ : ℤ, (5 * (-3) = Δ - 3) → (Δ = -12) := by
  sorry

end delta_value_l2171_217115


namespace factorization_proof_l2171_217190

theorem factorization_proof (x y : ℝ) : x * y^2 - 6 * x * y + 9 * x = x * (y - 3)^2 := by
  sorry

end factorization_proof_l2171_217190


namespace problem1_l2171_217100

theorem problem1 (a b : ℝ) (ha : a ≠ 0) :
  (a - b^2 / a) / ((a^2 + 2*a*b + b^2) / a) = (a - b) / (a + b) := by
  sorry

end problem1_l2171_217100


namespace duty_schedules_count_l2171_217147

/-- Represents the number of people on duty -/
def num_people : ℕ := 3

/-- Represents the number of days in the duty schedule -/
def num_days : ℕ := 6

/-- Represents the number of duty days per person -/
def duty_days_per_person : ℕ := 2

/-- Calculates the number of valid duty schedules -/
def count_duty_schedules : ℕ :=
  let total_arrangements := (num_days.choose duty_days_per_person) * ((num_days - duty_days_per_person).choose duty_days_per_person)
  let invalid_arrangements := 2 * ((num_days - 1).choose duty_days_per_person) * ((num_days - duty_days_per_person - 1).choose duty_days_per_person)
  let double_counted := ((num_days - 2).choose duty_days_per_person) * ((num_days - duty_days_per_person - 2).choose duty_days_per_person)
  total_arrangements - invalid_arrangements + double_counted

theorem duty_schedules_count :
  count_duty_schedules = 42 :=
sorry

end duty_schedules_count_l2171_217147


namespace third_roll_greater_probability_l2171_217189

def roll_count : ℕ := 3
def sides : ℕ := 8

def favorable_outcomes (sides : ℕ) : ℕ := 
  (sides - 1) * (sides - 1) + (sides - 1)

theorem third_roll_greater_probability (sides : ℕ) (h : sides > 0) :
  (favorable_outcomes sides : ℚ) / (sides ^ roll_count) = 7 / 64 :=
sorry

end third_roll_greater_probability_l2171_217189


namespace ned_weekly_earnings_l2171_217113

/-- Calculates weekly earnings from selling left-handed mice --/
def weekly_earnings (normal_price : ℝ) (price_increase_percent : ℝ) 
                    (daily_sales : ℕ) (open_days : ℕ) : ℝ :=
  let left_handed_price := normal_price * (1 + price_increase_percent)
  let daily_earnings := left_handed_price * daily_sales
  daily_earnings * open_days

/-- Theorem stating Ned's weekly earnings --/
theorem ned_weekly_earnings :
  weekly_earnings 120 0.3 25 4 = 15600 := by
  sorry

#eval weekly_earnings 120 0.3 25 4

end ned_weekly_earnings_l2171_217113


namespace polynomial_evaluation_l2171_217118

theorem polynomial_evaluation :
  ∀ x : ℝ, x > 0 → x^2 - 3*x - 9 = 0 → x^3 - 3*x^2 - 9*x + 27 = 27 := by
  sorry

end polynomial_evaluation_l2171_217118


namespace height_correction_percentage_l2171_217138

/-- Proves that given a candidate's actual height of 5 feet 8 inches (68 inches),
    and an initial overstatement of 25%, the percentage correction from the
    stated height to the actual height is 20%. -/
theorem height_correction_percentage (actual_height : ℝ) (stated_height : ℝ) :
  actual_height = 68 →
  stated_height = actual_height * 1.25 →
  (stated_height - actual_height) / stated_height * 100 = 20 := by
  sorry

end height_correction_percentage_l2171_217138


namespace mutually_exclusive_not_complementary_l2171_217188

-- Define the set of cards
inductive Card : Type
  | Hearts | Spades | Diamonds | Clubs

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "Person A gets a club"
def event_A_club (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "Person B gets a club"
def event_B_club (d : Distribution) : Prop := d Person.B = Card.Clubs

-- Theorem statement
theorem mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A_club d ∧ event_B_club d)) ∧
  (∃ d : Distribution, ¬event_A_club d ∧ ¬event_B_club d) :=
sorry

end mutually_exclusive_not_complementary_l2171_217188


namespace largest_non_prime_sequence_l2171_217191

theorem largest_non_prime_sequence : ∃ (a : ℕ), 
  (a ≥ 10 ∧ a + 6 ≤ 50) ∧ 
  (∀ i ∈ (Finset.range 7), ¬ Nat.Prime (a + i)) ∧
  (∀ b : ℕ, b > a + 6 → 
    ¬(b ≥ 10 ∧ b + 6 ≤ 50 ∧ 
      (∀ i ∈ (Finset.range 7), ¬ Nat.Prime (b + i)))) :=
by sorry

end largest_non_prime_sequence_l2171_217191


namespace two_dress_combinations_l2171_217183

def num_colors : Nat := 4
def num_patterns : Nat := 5

theorem two_dress_combinations : 
  (num_colors * num_patterns) * ((num_colors - 1) * (num_patterns - 1)) = 240 := by
  sorry

end two_dress_combinations_l2171_217183


namespace triangle_side_calculation_l2171_217198

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  2 * Real.sin (2 * B + π / 6) = 2 →
  a * c = 3 * Real.sqrt 3 →
  a + c = 4 →
  b ^ 2 = 16 - 9 * Real.sqrt 3 := by
  sorry

end triangle_side_calculation_l2171_217198


namespace count_valid_concatenations_eq_825957_l2171_217164

def is_valid_integer (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

def concatenate (a b c : ℕ) : ℕ := sorry

def count_valid_concatenations : ℕ := sorry

theorem count_valid_concatenations_eq_825957 :
  count_valid_concatenations = 825957 := by sorry

end count_valid_concatenations_eq_825957_l2171_217164
