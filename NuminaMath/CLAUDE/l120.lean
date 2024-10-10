import Mathlib

namespace shopkeeper_total_cards_l120_12032

-- Define the number of cards in a standard deck
def standard_deck_size : ℕ := 52

-- Define the number of complete decks the shopkeeper has
def complete_decks : ℕ := 3

-- Define the number of additional cards
def additional_cards : ℕ := 4

-- Theorem to prove
theorem shopkeeper_total_cards : 
  complete_decks * standard_deck_size + additional_cards = 160 := by
  sorry

end shopkeeper_total_cards_l120_12032


namespace pages_difference_l120_12084

/-- The number of pages Juwella read over four nights -/
def total_pages : ℕ := 100

/-- The number of pages Juwella will read tonight -/
def pages_tonight : ℕ := 20

/-- The number of pages Juwella read three nights ago -/
def pages_three_nights_ago : ℕ := 15

/-- The number of pages Juwella read two nights ago -/
def pages_two_nights_ago : ℕ := 2 * pages_three_nights_ago

/-- The number of pages Juwella read last night -/
def pages_last_night : ℕ := total_pages - pages_tonight - pages_three_nights_ago - pages_two_nights_ago

theorem pages_difference : pages_last_night - pages_two_nights_ago = 5 := by
  sorry

end pages_difference_l120_12084


namespace expression_value_at_three_l120_12034

theorem expression_value_at_three :
  let x : ℝ := 3
  (x^3 - 2*x^2 - 21*x + 36) / (x - 6) = 6 := by
  sorry

end expression_value_at_three_l120_12034


namespace some_multiplier_value_l120_12051

theorem some_multiplier_value : ∃ m : ℕ, (422 + 404)^2 - (m * 422 * 404) = 324 ∧ m = 4 := by
  sorry

end some_multiplier_value_l120_12051


namespace parallelogram_circles_theorem_l120_12047

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Checks if four points form a parallelogram -/
def is_parallelogram (p : Parallelogram) : Prop :=
  -- Add parallelogram conditions here
  sorry

/-- Checks if a circle passes through four points -/
def circle_passes_through (c : Circle) (p1 p2 p3 p4 : Point) : Prop :=
  -- Add circle condition here
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  -- Add distance calculation here
  sorry

theorem parallelogram_circles_theorem (ABCD : Parallelogram) (E F : Point) (ω1 ω2 : Circle) :
  is_parallelogram ABCD →
  distance ABCD.A ABCD.B > distance ABCD.B ABCD.C →
  circle_passes_through ω1 ABCD.A ABCD.D E F →
  circle_passes_through ω2 ABCD.B ABCD.C E F →
  ∃ (X Y : Point),
    distance ABCD.B X = 200 ∧
    distance X Y = 9 ∧
    distance Y ABCD.D = 80 →
    distance ABCD.B ABCD.C = 51 :=
sorry

end parallelogram_circles_theorem_l120_12047


namespace power_division_rule_l120_12057

theorem power_division_rule (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end power_division_rule_l120_12057


namespace multiples_of_four_l120_12024

theorem multiples_of_four (n : ℕ) : n = 16 ↔ 
  (∃ (l : List ℕ), l.length = 25 ∧ 
    (∀ m ∈ l, m % 4 = 0 ∧ n ≤ m ∧ m ≤ 112) ∧
    (∀ k : ℕ, n ≤ k ∧ k ≤ 112 ∧ k % 4 = 0 → k ∈ l) ∧
    (∀ m : ℕ, n < m → 
      ¬∃ (l' : List ℕ), l'.length = 25 ∧ 
        (∀ m' ∈ l', m' % 4 = 0 ∧ m ≤ m' ∧ m' ≤ 112) ∧
        (∀ k : ℕ, m ≤ k ∧ k ≤ 112 ∧ k % 4 = 0 → k ∈ l'))) :=
by sorry

end multiples_of_four_l120_12024


namespace max_sphere_area_from_cube_l120_12038

/-- The maximum surface area of a sphere carved from a cube -/
theorem max_sphere_area_from_cube (cube_side : ℝ) (sphere_radius : ℝ) : 
  cube_side = 2 →
  sphere_radius ≤ 1 →
  sphere_radius > 0 →
  (4 : ℝ) * Real.pi * sphere_radius^2 ≤ 4 * Real.pi :=
by
  sorry

#check max_sphere_area_from_cube

end max_sphere_area_from_cube_l120_12038


namespace cost_price_calculation_l120_12087

/-- Proves that given a selling price of 400 Rs. and a profit percentage of 25%, the cost price is 320 Rs. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 400 →
  profit_percentage = 25 →
  selling_price = (1 + profit_percentage / 100) * 320 :=
by
  sorry

#check cost_price_calculation

end cost_price_calculation_l120_12087


namespace sum_of_x_and_y_equals_four_l120_12085

theorem sum_of_x_and_y_equals_four (x y : ℝ) (i : ℂ) (h : i^2 = -1) 
  (eq : y + (2 - x) * i = 1 - i) : x + y = 4 := by
  sorry

end sum_of_x_and_y_equals_four_l120_12085


namespace ellipse_semi_minor_axis_l120_12075

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis,
    prove that its semi-minor axis has length √7 -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h1 : center = (-2, 1))
  (h2 : focus = (-3, 0))
  (h3 : semi_major_endpoint = (-2, 4)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 7 := by sorry

end ellipse_semi_minor_axis_l120_12075


namespace three_step_to_one_eleven_step_to_one_l120_12012

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 1

def reaches_one_in (n : ℕ) (steps : ℕ) : Prop :=
  ∃ (sequence : ℕ → ℕ), 
    sequence 0 = n ∧
    sequence steps = 1 ∧
    ∀ i < steps, sequence (i + 1) = operation (sequence i)

theorem three_step_to_one :
  ∃! (s : Finset ℕ), 
    s.card = 3 ∧ 
    ∀ n, n ∈ s ↔ reaches_one_in n 3 :=
sorry

theorem eleven_step_to_one :
  ∃! (s : Finset ℕ), 
    s.card = 3 ∧ 
    ∀ n, n ∈ s ↔ reaches_one_in n 11 :=
sorry

end three_step_to_one_eleven_step_to_one_l120_12012


namespace line_parameterization_l120_12090

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 7

/-- The parametric equation of the line -/
def parametric_equation (s n t x y : ℝ) : Prop :=
  x = s + 2 * t ∧ y = -3 + n * t

/-- The theorem stating the values of s and n -/
theorem line_parameterization :
  ∃ (s n : ℝ), (∀ (t x y : ℝ), parametric_equation s n t x y → line_equation x y) ∧ s = 2 ∧ n = 4 := by
  sorry

end line_parameterization_l120_12090


namespace positive_sum_greater_than_abs_difference_l120_12097

theorem positive_sum_greater_than_abs_difference (x y : ℝ) :
  x + y > |x - y| ↔ x > 0 ∧ y > 0 := by sorry

end positive_sum_greater_than_abs_difference_l120_12097


namespace gcd_three_numbers_l120_12001

theorem gcd_three_numbers (a b c : ℕ) (h1 : Nat.gcd a b = 18) (h2 : Nat.gcd b c = 18) :
  Nat.gcd a (Nat.gcd b c) = 18 := by
  sorry

end gcd_three_numbers_l120_12001


namespace kitten_growth_l120_12023

/-- The length of a kitten after doubling twice -/
def kitten_length (initial_length : ℝ) : ℝ :=
  initial_length * 2 * 2

/-- Theorem: A kitten with initial length 4 inches will be 16 inches long after doubling twice -/
theorem kitten_growth : kitten_length 4 = 16 := by
  sorry

end kitten_growth_l120_12023


namespace largest_two_digit_remainder_2_mod_13_l120_12014

theorem largest_two_digit_remainder_2_mod_13 :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 13 = 2 → n ≤ 93 :=
by sorry

end largest_two_digit_remainder_2_mod_13_l120_12014


namespace q_polynomial_form_l120_12055

/-- Given a function q(x) satisfying the equation
    q(x) + (2x^5 + 5x^4 + 8x^3 + 9x) = (10x^4 + 35x^3 + 50x^2 + 72x + 5),
    prove that q(x) = -2x^5 + 5x^4 + 27x^3 + 50x^2 + 63x + 5 -/
theorem q_polynomial_form (q : ℝ → ℝ) 
    (h : ∀ x, q x + (2*x^5 + 5*x^4 + 8*x^3 + 9*x) = 10*x^4 + 35*x^3 + 50*x^2 + 72*x + 5) :
  ∀ x, q x = -2*x^5 + 5*x^4 + 27*x^3 + 50*x^2 + 63*x + 5 := by
  sorry

end q_polynomial_form_l120_12055


namespace circle_area_theorem_l120_12043

theorem circle_area_theorem (z₁ z₂ : ℂ) 
  (h₁ : z₁^2 - 4*z₁*z₂ + 4*z₂^2 = 0) 
  (h₂ : Complex.abs z₂ = 2) : 
  Real.pi * (Complex.abs z₁ / 2)^2 = 4 * Real.pi :=
by sorry

end circle_area_theorem_l120_12043


namespace circle_and_trajectory_l120_12068

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line x - y + 1 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, -2)

-- Define point D
def D : ℝ × ℝ := (4, 3)

theorem circle_and_trajectory :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    C ∈ Line ∧
    A ∈ Circle C r ∧
    B ∈ Circle C r ∧
    (∀ (x y : ℝ), (x + 1)^2 + y^2 = 4 ↔ (x, y) ∈ Circle C r) ∧
    (∀ (x y : ℝ), (x - 1.5)^2 + (y - 1.5)^2 = 1 ↔
      ∃ (E : ℝ × ℝ), E ∈ Circle C r ∧ (x, y) = ((D.1 + E.1) / 2, (D.2 + E.2) / 2)) :=
by sorry

end circle_and_trajectory_l120_12068


namespace sum_mod_nine_l120_12044

theorem sum_mod_nine : (8150 + 8151 + 8152 + 8153 + 8154 + 8155) % 9 = 6 := by
  sorry

end sum_mod_nine_l120_12044


namespace expression_value_l120_12026

theorem expression_value :
  let a : ℤ := 3
  let b : ℤ := 7
  let c : ℤ := 2
  ((a * b - c) - (a + b * c)) - ((a * c - b) - (a - b * c)) = -8 := by
  sorry

end expression_value_l120_12026


namespace fraction_simplification_l120_12083

theorem fraction_simplification :
  (1/2 + 1/5) / (3/7 - 1/14) = 49/25 := by
  sorry

end fraction_simplification_l120_12083


namespace equation_solutions_l120_12053

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (3 * x₁^2 + 3 * x₁ + 6 = |(-20 + 5 * x₁)|) ∧ 
  (3 * x₂^2 + 3 * x₂ + 6 = |(-20 + 5 * x₂)|) ∧ 
  (x₁ ≠ x₂) ∧ 
  (-4 < x₁) ∧ (x₁ < 2) ∧ 
  (-4 < x₂) ∧ (x₂ < 2) := by
  sorry

end equation_solutions_l120_12053


namespace handshakes_in_specific_gathering_l120_12028

/-- Represents a gathering of people with specific knowledge relationships. -/
structure Gathering where
  total : Nat
  know_each_other : Nat
  know_no_one : Nat

/-- Calculates the number of handshakes in a gathering. -/
def count_handshakes (g : Gathering) : Nat :=
  g.know_no_one * (g.total - 1)

/-- Theorem stating that in a specific gathering, 217 handshakes occur. -/
theorem handshakes_in_specific_gathering :
  ∃ (g : Gathering),
    g.total = 30 ∧
    g.know_each_other = 15 ∧
    g.know_no_one = 15 ∧
    count_handshakes g = 217 := by
  sorry

#check handshakes_in_specific_gathering

end handshakes_in_specific_gathering_l120_12028


namespace square_angle_problem_l120_12006

/-- In a square ABCD with a segment CE, if two angles formed are 7α and 8α, then α = 9°. -/
theorem square_angle_problem (α : ℝ) : 
  (7 * α + 8 * α + 45 = 180) → α = 9 := by
  sorry

end square_angle_problem_l120_12006


namespace star_op_greater_star_op_commutative_l120_12078

-- Define the new operation ※ for rational numbers
def star_op (a b : ℚ) : ℚ := (a + b + abs (a - b)) / 2

-- Theorem for part (2)
theorem star_op_greater (a b : ℚ) (h : a > b) : star_op a b = a := by sorry

-- Theorem for part (3)
theorem star_op_commutative (a b : ℚ) : star_op a b = star_op b a := by sorry

-- Examples for part (1)
example : star_op 2 3 = 3 := by sorry
example : star_op 3 3 = 3 := by sorry
example : star_op (-2) (-3) = -2 := by sorry

end star_op_greater_star_op_commutative_l120_12078


namespace opposite_of_negative_two_l120_12003

theorem opposite_of_negative_two : (-(- 2) = 2) := by
  sorry

end opposite_of_negative_two_l120_12003


namespace problem_solution_l120_12062

theorem problem_solution (a b c : ℝ) : 
  (-(a) = -2) → 
  (1 / b = -3/2) → 
  (abs c = 2) → 
  a + b + c^2 = 16/3 := by
sorry

end problem_solution_l120_12062


namespace fraction_value_l120_12005

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  (a * c) / (b * d) = 20 := by
sorry

end fraction_value_l120_12005


namespace matrix_cube_computation_l120_12086

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube_computation :
  A ^ 3 = !![3, -6; 6, -3] := by sorry

end matrix_cube_computation_l120_12086


namespace find_number_l120_12002

theorem find_number (A B : ℕ+) (h1 : B = 286) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 26) : A = 210 := by
  sorry

end find_number_l120_12002


namespace product_is_real_product_is_imaginary_l120_12018

/-- The product of two complex numbers is real if and only if ad + bc = 0 -/
theorem product_is_real (a b c d : ℝ) :
  (Complex.I * b + a) * (Complex.I * d + c) ∈ Set.range (Complex.ofReal) ↔ a * d + b * c = 0 := by
  sorry

/-- The product of two complex numbers is purely imaginary if and only if ac - bd = 0 -/
theorem product_is_imaginary (a b c d : ℝ) :
  ∃ (k : ℝ), (Complex.I * b + a) * (Complex.I * d + c) = Complex.I * k ↔ a * c - b * d = 0 := by
  sorry

end product_is_real_product_is_imaginary_l120_12018


namespace hash_five_three_l120_12011

-- Define the # operation
def hash (a b : ℤ) : ℤ := 4 * a + 6 * b

-- Theorem statement
theorem hash_five_three : hash 5 3 = 38 := by
  sorry

end hash_five_three_l120_12011


namespace election_winner_percentage_l120_12058

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) : 
  total_votes = 480 → majority = 192 → 
  ∃ (winner_percentage : ℚ), 
    winner_percentage = 70 / 100 ∧ 
    (winner_percentage * total_votes : ℚ) - ((1 - winner_percentage) * total_votes : ℚ) = majority :=
by sorry

end election_winner_percentage_l120_12058


namespace quadratic_inequality_l120_12079

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end quadratic_inequality_l120_12079


namespace isosceles_triangle_angle_measure_l120_12060

theorem isosceles_triangle_angle_measure :
  ∀ (D E F : ℝ),
  -- Triangle DEF is isosceles with angle D congruent to angle F
  D = F →
  -- The measure of angle F is three times the measure of angle E
  F = 3 * E →
  -- The sum of angles in a triangle is 180 degrees
  D + E + F = 180 →
  -- The measure of angle D is 540/7 degrees
  D = 540 / 7 := by
sorry

end isosceles_triangle_angle_measure_l120_12060


namespace smallest_cube_root_with_small_remainder_l120_12004

theorem smallest_cube_root_with_small_remainder :
  ∃ (m : ℕ) (r : ℝ),
    (∀ (m' : ℕ) (r' : ℝ), m' < m → ¬(∃ (n' : ℕ), m'^(1/3 : ℝ) = n' + r' ∧ 0 < r' ∧ r' < 1/10000)) ∧
    m^(1/3 : ℝ) = 58 + r ∧
    0 < r ∧
    r < 1/10000 ∧
    (∀ (n : ℕ), n < 58 → 
      ¬(∃ (m' : ℕ) (r' : ℝ), m'^(1/3 : ℝ) = n + r' ∧ 0 < r' ∧ r' < 1/10000)) :=
sorry

end smallest_cube_root_with_small_remainder_l120_12004


namespace sum_of_triangles_l120_12081

-- Define the triangle operation
def triangle (a b c : ℤ) : ℤ := a + 2*b - c

-- Theorem statement
theorem sum_of_triangles : triangle 3 5 7 + triangle 6 1 8 = 6 := by
  sorry

end sum_of_triangles_l120_12081


namespace nickel_chocolates_l120_12008

theorem nickel_chocolates (robert : ℕ) (nickel : ℕ) 
  (h1 : robert = 10) 
  (h2 : robert = nickel + 5) : 
  nickel = 5 := by
  sorry

end nickel_chocolates_l120_12008


namespace number_puzzle_l120_12037

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 9) = 63 := by
  sorry

end number_puzzle_l120_12037


namespace sarah_trucks_l120_12031

theorem sarah_trucks (trucks_to_jeff trucks_to_ashley trucks_remaining : ℕ) 
  (h1 : trucks_to_jeff = 13)
  (h2 : trucks_to_ashley = 21)
  (h3 : trucks_remaining = 38) :
  trucks_to_jeff + trucks_to_ashley + trucks_remaining = 72 := by
  sorry

end sarah_trucks_l120_12031


namespace ellipse_and_hyperbola_properties_l120_12000

/-- An ellipse with foci on the y-axis -/
structure Ellipse where
  major_axis : ℝ
  minor_axis : ℝ
  foci_on_y_axis : Bool

/-- A hyperbola with foci on the y-axis -/
structure Hyperbola where
  real_axis : ℝ
  imaginary_axis : ℝ
  foci_on_y_axis : Bool

/-- Given ellipse properties, prove its equation, foci coordinates, eccentricity, and related hyperbola equation -/
theorem ellipse_and_hyperbola_properties (e : Ellipse) 
    (h1 : e.major_axis = 10) 
    (h2 : e.minor_axis = 8) 
    (h3 : e.foci_on_y_axis = true) : 
  (∃ (x y : ℝ), x^2/16 + y^2/25 = 1) ∧ 
  (∃ (f1 f2 : ℝ × ℝ), f1 = (0, -3) ∧ f2 = (0, 3)) ∧
  (3/5 : ℝ) = (5^2 - 4^2).sqrt / 5 ∧
  (∃ (h : Hyperbola), h.real_axis = 3 ∧ h.imaginary_axis = 4 ∧ h.foci_on_y_axis = true ∧
    ∃ (x y : ℝ), y^2/9 - x^2/16 = 1) :=
by sorry

end ellipse_and_hyperbola_properties_l120_12000


namespace lucy_shells_found_l120_12066

theorem lucy_shells_found (initial_shells final_shells : ℝ) 
  (h1 : initial_shells = 68.3)
  (h2 : final_shells = 89.5) :
  final_shells - initial_shells = 21.2 := by
  sorry

end lucy_shells_found_l120_12066


namespace one_carbon_per_sheet_l120_12050

/-- Represents the number of carbon copies produced when sheets are folded and typed on -/
def carbon_copies_when_folded : ℕ := 2

/-- Represents the total number of sheets -/
def total_sheets : ℕ := 3

/-- Represents the number of carbons in each sheet -/
def carbons_per_sheet : ℕ := 1

/-- Theorem stating that there is 1 carbon in each sheet -/
theorem one_carbon_per_sheet :
  (carbons_per_sheet = 1) ∧ 
  (carbon_copies_when_folded = 2) ∧
  (total_sheets = 3) := by
  sorry

end one_carbon_per_sheet_l120_12050


namespace f_composition_of_three_l120_12046

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_composition_of_three : f (f (f 3)) = 107 := by
  sorry

end f_composition_of_three_l120_12046


namespace cube_cutting_l120_12072

theorem cube_cutting (n : ℕ) : 
  (∃ s : ℕ, n > s ∧ n^3 - s^3 = 152) → n = 6 := by
  sorry

end cube_cutting_l120_12072


namespace john_tax_rate_l120_12089

/-- Given the number of shirts, price per shirt, and total payment including tax,
    calculate the tax rate as a percentage. -/
def calculate_tax_rate (num_shirts : ℕ) (price_per_shirt : ℚ) (total_payment : ℚ) : ℚ :=
  let cost_before_tax := num_shirts * price_per_shirt
  let tax_amount := total_payment - cost_before_tax
  (tax_amount / cost_before_tax) * 100

/-- Theorem stating that for 3 shirts at $20 each and a total payment of $66,
    the tax rate is 10%. -/
theorem john_tax_rate :
  calculate_tax_rate 3 20 66 = 10 := by
  sorry

#eval calculate_tax_rate 3 20 66

end john_tax_rate_l120_12089


namespace complex_number_equality_l120_12027

theorem complex_number_equality (z : ℂ) :
  Complex.abs (z - 2) = 5 ∧ 
  Complex.abs (z + 4) = 5 ∧ 
  Complex.abs (z - 2*I) = 5 → 
  z = -1 - 4*I :=
by sorry

end complex_number_equality_l120_12027


namespace algebraic_expression_value_l120_12071

theorem algebraic_expression_value (a b : ℝ) (h : 3 * a * b - 3 * b^2 - 2 = 0) :
  (1 - (2 * a * b - b^2) / a^2) / ((a - b) / (a^2 * b)) = 2/3 := by
  sorry

end algebraic_expression_value_l120_12071


namespace max_discount_rate_l120_12061

/-- The maximum discount rate that can be offered on an item while maintaining a minimum profit margin -/
theorem max_discount_rate (cost_price selling_price min_profit_margin : ℝ) 
  (h1 : cost_price = 4)
  (h2 : selling_price = 5)
  (h3 : min_profit_margin = 0.1)
  (h4 : cost_price > 0)
  (h5 : selling_price > cost_price) :
  ∃ (max_discount : ℝ), 
    max_discount = 12 ∧ 
    ∀ (discount : ℝ), 
      0 ≤ discount → discount ≤ max_discount → 
      (selling_price * (1 - discount / 100) - cost_price) / cost_price ≥ min_profit_margin :=
by sorry

end max_discount_rate_l120_12061


namespace polynomial_factorization_l120_12098

theorem polynomial_factorization (x : ℤ) :
  x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) :=
by sorry

end polynomial_factorization_l120_12098


namespace apples_picked_total_l120_12067

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_picked_total :
  total_apples = 11 :=
by sorry

end apples_picked_total_l120_12067


namespace arithmetic_seq_2015th_term_l120_12093

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_1_eq_1 : a 1 = 1
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  d_neq_0 : a 2 - a 1 ≠ 0
  geometric_subseq : (a 2)^2 = a 1 * a 5

/-- The 2015th term of the arithmetic sequence is 4029 -/
theorem arithmetic_seq_2015th_term (seq : ArithmeticSequence) : seq.a 2015 = 4029 := by
  sorry

end arithmetic_seq_2015th_term_l120_12093


namespace units_digit_factorial_sum_100_l120_12035

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_100 :
  units_digit (factorial_sum 100) = 3 := by
sorry

end units_digit_factorial_sum_100_l120_12035


namespace exists_prime_number_of_ones_l120_12052

/-- A number consisting of q ones in decimal notation -/
def number_of_ones (q : ℕ) : ℕ := (10^q - 1) / 9

/-- Theorem stating that there exists a natural number k such that
    a number consisting of (6k-1) ones is prime -/
theorem exists_prime_number_of_ones :
  ∃ k : ℕ, Nat.Prime (number_of_ones (6*k - 1)) := by
  sorry

end exists_prime_number_of_ones_l120_12052


namespace x_cube_plus_four_x_equals_eight_l120_12009

theorem x_cube_plus_four_x_equals_eight (x : ℝ) (h : x^3 + 4*x = 8) :
  x^7 + 64*x^2 = 128 := by
sorry

end x_cube_plus_four_x_equals_eight_l120_12009


namespace combined_average_score_l120_12042

/-- Given two math modeling clubs with their respective member counts and average scores,
    calculate the combined average score of both clubs. -/
theorem combined_average_score
  (club_a_members : ℕ)
  (club_b_members : ℕ)
  (club_a_average : ℝ)
  (club_b_average : ℝ)
  (h1 : club_a_members = 40)
  (h2 : club_b_members = 50)
  (h3 : club_a_average = 90)
  (h4 : club_b_average = 81) :
  (club_a_members * club_a_average + club_b_members * club_b_average) /
  (club_a_members + club_b_members : ℝ) = 85 := by
  sorry

end combined_average_score_l120_12042


namespace square_root_division_l120_12033

theorem square_root_division (x : ℝ) : (Real.sqrt 3600 / x = 4) → x = 15 := by
  sorry

end square_root_division_l120_12033


namespace special_line_equation_l120_12036

/-- A line passing through the point (3, -4) with intercepts on the coordinate axes that are opposite numbers -/
structure SpecialLine where
  /-- The equation of the line in the form ax + by + c = 0 -/
  equation : ℝ → ℝ → ℝ → ℝ
  /-- The line passes through the point (3, -4) -/
  passes_through_point : equation 3 (-4) 0 = 0
  /-- The x-intercept and y-intercept are opposite numbers -/
  opposite_intercepts : ∃ (a : ℝ), equation a 0 0 = 0 ∧ equation 0 (-a) 0 = 0

/-- The equation of the special line is either 4x + 3y = 0 or x - y - 7 = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, l.equation x y 0 = 4*x + 3*y) ∨
  (∀ x y, l.equation x y 0 = x - y - 7) :=
sorry

end special_line_equation_l120_12036


namespace sum_ge_sum_of_abs_div_three_l120_12040

theorem sum_ge_sum_of_abs_div_three (a b c : ℝ) 
  (hab : a + b ≥ 0) (hbc : b + c ≥ 0) (hca : c + a ≥ 0) :
  a + b + c ≥ (|a| + |b| + |c|) / 3 := by
  sorry

end sum_ge_sum_of_abs_div_three_l120_12040


namespace polynomial_derivative_symmetry_l120_12063

/-- Given a polynomial function f(x) = ax^4 + bx^2 + c, 
    if f'(1) = 2, then f'(-1) = -2 -/
theorem polynomial_derivative_symmetry 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^4 + b * x^2 + c)
  (h_f'_1 : (deriv f) 1 = 2) : 
  (deriv f) (-1) = -2 := by
sorry

end polynomial_derivative_symmetry_l120_12063


namespace sum_of_eight_five_to_eight_l120_12099

theorem sum_of_eight_five_to_eight (n : ℕ) :
  (Finset.range 8).sum (λ _ => 5^8) = 3125000 := by
  sorry

end sum_of_eight_five_to_eight_l120_12099


namespace line_passes_through_point_l120_12096

/-- Given a line with equation 2 + 3kx = -7y that passes through the point (-1/3, 4),
    prove that k = 30. -/
theorem line_passes_through_point (k : ℝ) : 
  (2 + 3 * k * (-1/3) = -7 * 4) → k = 30 := by
  sorry

end line_passes_through_point_l120_12096


namespace min_students_in_class_l120_12092

theorem min_students_in_class (boys girls : ℕ) : 
  boys > 0 → 
  girls > 0 → 
  (3 * boys) % 4 = 0 → 
  (3 * boys) / 4 = girls / 2 → 
  5 ≤ boys + girls :=
sorry

end min_students_in_class_l120_12092


namespace blue_shirts_count_l120_12015

theorem blue_shirts_count (total_shirts green_shirts : ℕ) 
  (h1 : total_shirts = 23)
  (h2 : green_shirts = 17)
  (h3 : total_shirts = green_shirts + blue_shirts) :
  blue_shirts = 6 :=
by
  sorry

end blue_shirts_count_l120_12015


namespace select_four_from_eighteen_l120_12054

theorem select_four_from_eighteen (n m : ℕ) : n = 18 ∧ m = 4 → Nat.choose n m = 3060 := by
  sorry

end select_four_from_eighteen_l120_12054


namespace city_rentals_per_mile_rate_l120_12030

/-- Represents the daily rental rate in dollars -/
def daily_rate_sunshine : ℝ := 17.99

/-- Represents the per-mile rate for Sunshine Car Rentals in dollars -/
def per_mile_rate_sunshine : ℝ := 0.18

/-- Represents the daily rental rate for City Rentals in dollars -/
def daily_rate_city : ℝ := 18.95

/-- Represents the number of miles driven -/
def miles_driven : ℝ := 48

/-- Represents the unknown per-mile rate for City Rentals -/
def per_mile_rate_city : ℝ := 0.16

theorem city_rentals_per_mile_rate :
  daily_rate_sunshine + per_mile_rate_sunshine * miles_driven =
  daily_rate_city + per_mile_rate_city * miles_driven :=
by sorry

#check city_rentals_per_mile_rate

end city_rentals_per_mile_rate_l120_12030


namespace power_division_equality_l120_12021

theorem power_division_equality : (3 : ℕ)^12 / (9 : ℕ)^2 = 6561 := by sorry

end power_division_equality_l120_12021


namespace min_value_expression_l120_12082

theorem min_value_expression (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  m + (n^2 - m*n + 4) / (m - n) ≥ 4 ∧
  (m + (n^2 - m*n + 4) / (m - n) = 4 ↔ m - n = 2) :=
by sorry

end min_value_expression_l120_12082


namespace stone_slab_length_l120_12080

/-- Given a floor covered by square stone slabs, this theorem calculates the length of each slab. -/
theorem stone_slab_length
  (num_slabs : ℕ)
  (total_area : ℝ)
  (h_num_slabs : num_slabs = 30)
  (h_total_area : total_area = 50.7) :
  ∃ (slab_length : ℝ),
    slab_length = 130 ∧
    num_slabs * (slab_length / 100)^2 = total_area :=
by sorry

end stone_slab_length_l120_12080


namespace fermat_number_divisibility_l120_12022

theorem fermat_number_divisibility (m n : ℕ) (h : m > n) :
  ∃ k : ℕ, 2^(2^m) - 1 = (2^(2^n) + 1) * k := by
  sorry

end fermat_number_divisibility_l120_12022


namespace shaded_area_square_with_quarter_circles_l120_12074

/-- The area of the shaded region in a square with quarter circles at corners -/
theorem shaded_area_square_with_quarter_circles 
  (side_length : ℝ) 
  (radius : ℝ) 
  (h1 : side_length = 12) 
  (h2 : radius = 6) : 
  side_length ^ 2 - π * radius ^ 2 = 144 - 36 * π := by
  sorry

#check shaded_area_square_with_quarter_circles

end shaded_area_square_with_quarter_circles_l120_12074


namespace rectangle_area_increase_l120_12069

theorem rectangle_area_increase (l w : ℝ) (h1 : l > 0) (h2 : w > 0) :
  (1.15 * l) * (1.25 * w) = 1.4375 * (l * w) := by sorry

end rectangle_area_increase_l120_12069


namespace greatest_integer_for_fraction_l120_12045

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem greatest_integer_for_fraction : 
  (∀ x : ℤ, x > 14 → ¬is_integer ((x^2 - 5*x + 14) / (x - 4))) ∧
  is_integer ((14^2 - 5*14 + 14) / (14 - 4)) := by
  sorry

end greatest_integer_for_fraction_l120_12045


namespace unique_sums_count_l120_12095

def bag_C : Finset ℕ := {1, 2, 3, 4}
def bag_D : Finset ℕ := {3, 5, 7}

theorem unique_sums_count : 
  Finset.card ((bag_C.product bag_D).image (fun (p : ℕ × ℕ) => p.1 + p.2)) = 8 := by
  sorry

end unique_sums_count_l120_12095


namespace difference_in_rubber_bands_l120_12016

-- Define the number of rubber bands Harper has
def harper_bands : ℕ := 15

-- Define the total number of rubber bands they have together
def total_bands : ℕ := 24

-- Define the number of rubber bands Harper's brother has
def brother_bands : ℕ := total_bands - harper_bands

-- Theorem to prove
theorem difference_in_rubber_bands :
  harper_bands - brother_bands = 6 ∧ brother_bands < harper_bands :=
by sorry

end difference_in_rubber_bands_l120_12016


namespace convex_ngon_angle_theorem_l120_12020

theorem convex_ngon_angle_theorem (n : ℕ) : 
  n ≥ 3 →  -- n-gon must have at least 3 sides
  (∃ (x : ℝ), x > 0 ∧ x < 150 ∧ 150 * (n - 1) + x = 180 * (n - 2)) →
  (n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11) :=
by sorry

end convex_ngon_angle_theorem_l120_12020


namespace amanda_kitchen_upgrade_l120_12088

/-- The total cost of Amanda's kitchen upgrade after applying discounts -/
def kitchen_upgrade_cost (cabinet_knobs : ℕ) (knob_price : ℚ) (drawer_pulls : ℕ) (pull_price : ℚ) 
  (knob_discount : ℚ) (pull_discount : ℚ) : ℚ :=
  let knob_total := cabinet_knobs * knob_price
  let pull_total := drawer_pulls * pull_price
  let discounted_knob_total := knob_total * (1 - knob_discount)
  let discounted_pull_total := pull_total * (1 - pull_discount)
  discounted_knob_total + discounted_pull_total

/-- Amanda's kitchen upgrade cost is $67.70 -/
theorem amanda_kitchen_upgrade : 
  kitchen_upgrade_cost 18 (5/2) 8 4 (1/10) (3/20) = 677/10 := by
  sorry

end amanda_kitchen_upgrade_l120_12088


namespace binary_sum_theorem_l120_12064

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits -/
def binary_number (bits : List Bool) : Nat := binary_to_decimal bits

theorem binary_sum_theorem :
  let a := binary_number [true, false, true, false, true]
  let b := binary_number [true, true, true]
  let c := binary_number [true, false, true, true, true, false]
  let d := binary_number [true, false, true, false, true, true]
  let sum := binary_number [true, true, true, true, false, false, true]
  a + b + c + d = sum := by sorry

end binary_sum_theorem_l120_12064


namespace inverse_of_B_cubed_l120_12039

/-- Given a 2x2 matrix B with its inverse, prove that the inverse of B^3 is as stated. -/
theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = ![![3, 4], ![-2, -2]]) : 
  (B^3)⁻¹ = ![![3, 4], ![-6, -28]] := by
  sorry

end inverse_of_B_cubed_l120_12039


namespace dot_product_a_b_l120_12048

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![3, -2]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Theorem statement
theorem dot_product_a_b : dot_product a b = 4 := by sorry

end dot_product_a_b_l120_12048


namespace specific_arithmetic_series_sum_l120_12041

/-- The sum of an arithmetic series with given first term, last term, and common difference -/
def arithmetic_series_sum (a l d : ℤ) : ℤ :=
  let n : ℤ := (l - a) / d + 1
  (n * (a + l)) / 2

/-- Theorem stating that the sum of the specific arithmetic series is -576 -/
theorem specific_arithmetic_series_sum :
  arithmetic_series_sum (-47) (-1) 2 = -576 := by
  sorry

end specific_arithmetic_series_sum_l120_12041


namespace trigonometric_expression_equality_l120_12056

theorem trigonometric_expression_equality : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end trigonometric_expression_equality_l120_12056


namespace sofa_loveseat_ratio_l120_12076

theorem sofa_loveseat_ratio (total_cost love_seat_cost sofa_cost : ℚ) : 
  total_cost = 444 →
  love_seat_cost = 148 →
  total_cost = sofa_cost + love_seat_cost →
  sofa_cost / love_seat_cost = 2 := by
sorry

end sofa_loveseat_ratio_l120_12076


namespace smallest_k_for_15_digit_period_l120_12059

/-- Represents a positive rational number with a decimal representation having a period of 30 digits -/
def RationalWith30DigitPeriod : Type := { q : ℚ // q > 0 ∧ ∃ m : ℕ, q = m / (10^30 - 1) }

/-- The theorem statement -/
theorem smallest_k_for_15_digit_period 
  (a b : RationalWith30DigitPeriod)
  (h_diff : ∃ p : ℤ, (a.val - b.val : ℚ) = p / (10^15 - 1)) :
  (∃ k : ℕ, k > 0 ∧ ∃ q : ℤ, (a.val + k * b.val : ℚ) = q / (10^15 - 1)) ∧
  (∀ k : ℕ, k > 0 → k < 6 → ¬∃ q : ℤ, (a.val + k * b.val : ℚ) = q / (10^15 - 1)) :=
sorry

end smallest_k_for_15_digit_period_l120_12059


namespace cubic_one_real_root_l120_12007

/-- A cubic equation with coefficients a and b -/
def cubic_equation (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- Condition for the cubic equation to have only one real root -/
def has_one_real_root (a b : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation a b x = 0

theorem cubic_one_real_root :
  (has_one_real_root (-3) (-3)) ∧
  (∀ b > 2, has_one_real_root (-3) b) ∧
  (has_one_real_root 0 2) :=
sorry

end cubic_one_real_root_l120_12007


namespace complementary_event_of_A_l120_12065

-- Define the sample space for a fair cubic die
def DieOutcome := Fin 6

-- Define event A
def EventA (outcome : DieOutcome) : Prop := outcome.val % 2 = 1

-- Define the complementary event of A
def ComplementA (outcome : DieOutcome) : Prop := outcome.val % 2 = 0

-- Theorem statement
theorem complementary_event_of_A :
  ∀ (outcome : DieOutcome), ¬(EventA outcome) ↔ ComplementA outcome :=
by sorry

end complementary_event_of_A_l120_12065


namespace part_one_part_two_l120_12094

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 3
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Part I
theorem part_one :
  let S := {x : ℝ | (p x ∨ q x 2) ∧ ¬(p x ∧ q x 2)}
  S = {x : ℝ | -4 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} :=
sorry

-- Part II
theorem part_two :
  let T := {m : ℝ | m > 0 ∧ {x : ℝ | p x} ⊃ {x : ℝ | q x m} ∧ {x : ℝ | p x} ≠ {x : ℝ | q x m}}
  T = {m : ℝ | 0 < m ∧ m ≤ 1} :=
sorry

end part_one_part_two_l120_12094


namespace range_of_a_l120_12049

def A (a : ℝ) : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≥ 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a - 1}

theorem range_of_a (a : ℝ) (h : A a ∪ B a = Set.univ) : a ≤ 2 := by
  sorry

#check range_of_a

end range_of_a_l120_12049


namespace stratified_sampling_size_l120_12091

theorem stratified_sampling_size 
  (high_school_students : ℕ) 
  (junior_high_students : ℕ) 
  (sampled_high_school : ℕ) 
  (h1 : high_school_students = 3500)
  (h2 : junior_high_students = 1500)
  (h3 : sampled_high_school = 70) :
  let total_students := high_school_students + junior_high_students
  let sampling_ratio := sampled_high_school / high_school_students
  let total_sample_size := total_students * sampling_ratio
  total_sample_size = 100 := by
sorry

end stratified_sampling_size_l120_12091


namespace milk_water_mixture_volume_l120_12019

theorem milk_water_mixture_volume 
  (initial_milk_percentage : Real)
  (final_milk_percentage : Real)
  (added_water : Real)
  (h1 : initial_milk_percentage = 0.84)
  (h2 : final_milk_percentage = 0.60)
  (h3 : added_water = 24)
  : ∃ initial_volume : Real,
    initial_volume * initial_milk_percentage = 
    (initial_volume + added_water) * final_milk_percentage ∧
    initial_volume = 60 := by
  sorry

end milk_water_mixture_volume_l120_12019


namespace simplify_fraction_l120_12010

theorem simplify_fraction (a : ℚ) (h : a = 2) : 15 * a^4 / (45 * a^3) = 2/3 := by
  sorry

end simplify_fraction_l120_12010


namespace cloth_sale_calculation_l120_12017

/-- Represents the number of meters of cloth sold -/
def meters_sold : ℕ := 30

/-- The total selling price in Rupees -/
def total_selling_price : ℕ := 4500

/-- The profit per meter in Rupees -/
def profit_per_meter : ℕ := 10

/-- The cost price per meter in Rupees -/
def cost_price_per_meter : ℕ := 140

/-- Theorem stating that the number of meters sold is correct given the conditions -/
theorem cloth_sale_calculation :
  meters_sold * (cost_price_per_meter + profit_per_meter) = total_selling_price :=
by sorry

end cloth_sale_calculation_l120_12017


namespace polynomial_equation_solution_l120_12013

theorem polynomial_equation_solution (a a1 a2 a3 a4 : ℝ) : 
  (∀ x, (x + a)^4 = x^4 + a1*x^3 + a2*x^2 + a3*x + a4) →
  (a1 + a2 + a3 = 64) →
  (a = 2) := by
sorry

end polynomial_equation_solution_l120_12013


namespace parallel_vectors_x_value_l120_12077

theorem parallel_vectors_x_value (x : ℝ) (a b : ℝ × ℝ) :
  a = (2, x) →
  b = (4, -1) →
  (∃ (k : ℝ), a = k • b) →
  x = -1/2 := by
sorry

end parallel_vectors_x_value_l120_12077


namespace quadratic_function_properties_l120_12029

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h1 : quadratic_function f) 
  (h2 : f 0 = 1) 
  (h3 : ∀ x, f (x + 1) - f x = 2 * x) :
  (∀ x, f x = x^2 - x + 1) ∧ 
  Set.Icc (3/4 : ℝ) 3 = {y | ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = y} :=
by sorry

end quadratic_function_properties_l120_12029


namespace product_of_fractions_l120_12073

theorem product_of_fractions : (1 / 3) * (1 / 4) * (1 / 5) * (1 / 6) = 1 / 360 := by
  sorry

end product_of_fractions_l120_12073


namespace min_distance_sum_l120_12025

/-- Given points M(-1,3) and N(2,1), and point P on the x-axis,
    the minimum value of PM+PN is 5. -/
theorem min_distance_sum (M N P : ℝ × ℝ) : 
  M = (-1, 3) → 
  N = (2, 1) → 
  P.2 = 0 → 
  ∃ (min_val : ℝ), (∀ Q : ℝ × ℝ, Q.2 = 0 → 
    Real.sqrt ((Q.1 - M.1)^2 + (Q.2 - M.2)^2) + 
    Real.sqrt ((Q.1 - N.1)^2 + (Q.2 - N.2)^2) ≥ min_val) ∧ 
  min_val = 5 := by
  sorry

end min_distance_sum_l120_12025


namespace ediths_books_l120_12070

/-- The total number of books Edith has, given the number of novels and their relation to writing books -/
theorem ediths_books (novels : ℕ) (writing_books : ℕ) 
  (h1 : novels = 80) 
  (h2 : novels = writing_books / 2) : 
  novels + writing_books = 240 := by
  sorry

end ediths_books_l120_12070
