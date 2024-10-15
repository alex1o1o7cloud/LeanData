import Mathlib

namespace NUMINAMATH_CALUDE_train_length_l2155_215595

/-- Proves that a train traveling at 40 km/hr crossing a pole in 9 seconds has a length of 100 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 40 → -- speed in km/hr
  time = 9 → -- time in seconds
  length = speed * (1000 / 3600) * time → -- convert km/hr to m/s and multiply by time
  length = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2155_215595


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2155_215559

/-- A line passing through (-1, 2) and perpendicular to y = 2/3x has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  let l : Set (ℝ × ℝ) := {(x, y) | 3 * x + 2 * y - 1 = 0}
  let point : ℝ × ℝ := (-1, 2)
  let perpendicular_slope : ℝ := 2 / 3
  (point ∈ l) ∧
  (∀ (x y : ℝ), (x, y) ∈ l → (3 : ℝ) * perpendicular_slope = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2155_215559


namespace NUMINAMATH_CALUDE_conference_handshakes_l2155_215570

/-- The number of handshakes in a conference of n people where each person
    shakes hands exactly once with every other person. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference of 12 people where each person shakes hands
    exactly once with every other person, there are 66 handshakes. -/
theorem conference_handshakes :
  handshakes 12 = 66 := by
  sorry

#eval handshakes 12

end NUMINAMATH_CALUDE_conference_handshakes_l2155_215570


namespace NUMINAMATH_CALUDE_sum_in_base_8_l2155_215526

/-- Converts a decimal number to its octal (base 8) representation -/
def toOctal (n : ℕ) : List ℕ :=
  sorry

/-- Converts an octal (base 8) representation to its decimal value -/
def fromOctal (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in their octal representations -/
def octalAdd (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_in_base_8 :
  let a := 624
  let b := 112
  let expected_sum := [1, 3, 4, 0]
  octalAdd (toOctal a) (toOctal b) = expected_sum ∧
  fromOctal expected_sum = a + b :=
by sorry

end NUMINAMATH_CALUDE_sum_in_base_8_l2155_215526


namespace NUMINAMATH_CALUDE_certain_number_value_certain_number_value_proof_l2155_215522

theorem certain_number_value : ℝ → Prop :=
  fun y =>
    let x : ℝ := (390 - (48 + 62 + 98 + 124)) -- x from the second set
    let first_set : List ℝ := [28, x, 42, 78, y]
    let second_set : List ℝ := [48, 62, 98, 124, x]
    (List.sum first_set / first_set.length = 62) ∧
    (List.sum second_set / second_set.length = 78) →
    y = 104

-- The proof goes here
theorem certain_number_value_proof : certain_number_value 104 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_certain_number_value_proof_l2155_215522


namespace NUMINAMATH_CALUDE_people_in_house_l2155_215574

theorem people_in_house : 
  ∀ (initial_bedroom : ℕ) (entering_bedroom : ℕ) (living_room : ℕ),
    initial_bedroom = 2 →
    entering_bedroom = 5 →
    living_room = 8 →
    initial_bedroom + entering_bedroom + living_room = 14 := by
  sorry

end NUMINAMATH_CALUDE_people_in_house_l2155_215574


namespace NUMINAMATH_CALUDE_x_coordinate_difference_l2155_215538

theorem x_coordinate_difference (m n k : ℝ) : 
  (m = 2*n + 5) → 
  (m + k = 2*(n + 2) + 5) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_x_coordinate_difference_l2155_215538


namespace NUMINAMATH_CALUDE_ab_equals_two_l2155_215516

theorem ab_equals_two (a b : ℝ) (h : (a + 1)^2 + (b + 2)^2 = 0) : a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_two_l2155_215516


namespace NUMINAMATH_CALUDE_equation_solution_l2155_215556

theorem equation_solution (y : ℝ) (h : y ≠ 0) :
  (3 / y - (4 / y) * (2 / y) = 1.5) → y = 1 + Real.sqrt (19 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2155_215556


namespace NUMINAMATH_CALUDE_biology_magnet_combinations_l2155_215588

def word : String := "BIOLOGY"

def num_vowels : Nat := 3
def num_consonants : Nat := 2
def num_Os : Nat := 2

def vowels : Finset Char := {'I', 'O'}
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

theorem biology_magnet_combinations : 
  (Finset.card (Finset.powerset vowels) * Finset.card (Finset.powerset consonants)) +
  (Finset.card (Finset.powerset {0, 1}) * Finset.card (Finset.powerset consonants)) = 42 := by
  sorry

end NUMINAMATH_CALUDE_biology_magnet_combinations_l2155_215588


namespace NUMINAMATH_CALUDE_sum_of_amp_operations_l2155_215510

-- Define the operation &
def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

-- Theorem statement
theorem sum_of_amp_operations : amp 12 5 + amp 8 3 = 174 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_amp_operations_l2155_215510


namespace NUMINAMATH_CALUDE_complex_multiplication_simplification_l2155_215552

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem: For any real number t, (2+t i)(2-t i) = 4 + t^2 -/
theorem complex_multiplication_simplification (t : ℝ) : 
  (2 + t * i) * (2 - t * i) = (4 : ℂ) + t^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_simplification_l2155_215552


namespace NUMINAMATH_CALUDE_simplify_product_of_radicals_l2155_215557

theorem simplify_product_of_radicals (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x) = 84 * x * Real.sqrt (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_product_of_radicals_l2155_215557


namespace NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l2155_215534

/-- The number of FGH supermarkets in the US, given the total number of supermarkets
    and the difference between US and Canadian supermarkets. -/
def us_supermarkets (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

theorem fgh_supermarkets_in_us :
  us_supermarkets 60 22 = 41 := by
  sorry

#eval us_supermarkets 60 22

end NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l2155_215534


namespace NUMINAMATH_CALUDE_solutions_to_equation_l2155_215566

def solution_set : Set ℂ := {
  (3 * Real.sqrt 2) / 2 + (3 * Real.sqrt 2) / 2 * Complex.I,
  -(3 * Real.sqrt 2) / 2 - (3 * Real.sqrt 2) / 2 * Complex.I,
  (3 * Real.sqrt 2) / 2 * Complex.I - (3 * Real.sqrt 2) / 2,
  -(3 * Real.sqrt 2) / 2 * Complex.I + (3 * Real.sqrt 2) / 2
}

theorem solutions_to_equation : 
  ∀ x : ℂ, x^4 + 81 = 0 ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_solutions_to_equation_l2155_215566


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2155_215575

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 9 ∧ x ≠ -6 →
  (4 * x - 3) / (x^2 - 3*x - 54) = (11/5) / (x - 9) + (9/5) / (x + 6) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2155_215575


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2155_215513

theorem least_subtraction_for_divisibility :
  ∃! r : ℕ, r < 47 ∧ (3674958423 - r) % 47 = 0 ∧ ∀ s : ℕ, s < r → (3674958423 - s) % 47 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2155_215513


namespace NUMINAMATH_CALUDE_find_number_l2155_215561

theorem find_number : ∃! x : ℝ, ((x * 2) - 37 + 25) / 8 = 5 := by sorry

end NUMINAMATH_CALUDE_find_number_l2155_215561


namespace NUMINAMATH_CALUDE_round_trip_no_car_percentage_l2155_215504

theorem round_trip_no_car_percentage
  (total_round_trip : ℝ)
  (round_trip_with_car : ℝ)
  (h1 : round_trip_with_car = 25)
  (h2 : total_round_trip = 62.5) :
  total_round_trip - round_trip_with_car = 37.5 := by
sorry

end NUMINAMATH_CALUDE_round_trip_no_car_percentage_l2155_215504


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2155_215547

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) :
  n > 2 →
  exterior_angle = 40 →
  (360 : ℝ) / exterior_angle = n →
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2155_215547


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l2155_215532

theorem max_xy_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 7 * x + 8 * y = 112) :
  x * y ≤ 56 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 7 * x₀ + 8 * y₀ = 112 ∧ x₀ * y₀ = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l2155_215532


namespace NUMINAMATH_CALUDE_five_student_committees_l2155_215576

theorem five_student_committees (n : ℕ) (k : ℕ) : n = 8 ∧ k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committees_l2155_215576


namespace NUMINAMATH_CALUDE_quarters_fraction_l2155_215599

/-- The number of state quarters in Stephanie's collection -/
def total_quarters : ℕ := 25

/-- The number of states that joined the union from 1800 to 1809 -/
def states_1800_1809 : ℕ := 8

/-- The fraction of quarters representing states that joined from 1800 to 1809 -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_quarters

theorem quarters_fraction :
  fraction_1800_1809 = 8 / 25 := by sorry

end NUMINAMATH_CALUDE_quarters_fraction_l2155_215599


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2155_215564

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (h : ℝ) 
  (lateral_area : ℝ) 
  (h_r : r = 3) 
  (h_h : h = 1) :
  lateral_area = 3 * Real.sqrt 10 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2155_215564


namespace NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l2155_215509

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem third_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 8) :
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_sequence_l2155_215509


namespace NUMINAMATH_CALUDE_sum_of_cubes_negative_l2155_215517

theorem sum_of_cubes_negative : 
  (Real.sqrt 2021 - Real.sqrt 2020)^3 + 
  (Real.sqrt 2020 - Real.sqrt 2019)^3 + 
  (Real.sqrt 2019 - Real.sqrt 2018)^3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_negative_l2155_215517


namespace NUMINAMATH_CALUDE_correct_factorization_l2155_215537

theorem correct_factorization (m : ℤ) : m^3 + m = m * (m^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l2155_215537


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2155_215577

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 12) (h2 : x + 4 * y = 16) : 
  17 * x^2 + 18 * x * y + 17 * y^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2155_215577


namespace NUMINAMATH_CALUDE_max_f_and_min_side_a_l2155_215579

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

theorem max_f_and_min_side_a :
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ 2) ∧
  (∀ (A B C a b c : ℝ),
    0 < A ∧ A < Real.pi →
    0 < B ∧ B < Real.pi →
    0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    f (B + C) = 3 / 2 →
    b + c = 2 →
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
    a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_f_and_min_side_a_l2155_215579


namespace NUMINAMATH_CALUDE_brand_preference_ratio_l2155_215511

/-- Given a survey with 250 total respondents and 200 preferring brand X,
    prove that the ratio of people preferring brand X to those preferring brand Y is 4:1 -/
theorem brand_preference_ratio (total : ℕ) (brand_x : ℕ) (h1 : total = 250) (h2 : brand_x = 200) :
  (brand_x : ℚ) / (total - brand_x : ℚ) = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_ratio_l2155_215511


namespace NUMINAMATH_CALUDE_mom_initial_money_l2155_215546

/-- The amount of money Mom spent on bananas -/
def banana_cost : ℕ := 2 * 4

/-- The amount of money Mom spent on pears -/
def pear_cost : ℕ := 2

/-- The amount of money Mom spent on asparagus -/
def asparagus_cost : ℕ := 6

/-- The amount of money Mom spent on chicken -/
def chicken_cost : ℕ := 11

/-- The amount of money Mom has left after shopping -/
def money_left : ℕ := 28

/-- The total amount Mom spent on groceries -/
def total_spent : ℕ := banana_cost + pear_cost + asparagus_cost + chicken_cost

/-- Theorem stating that Mom had €55 when she left for the market -/
theorem mom_initial_money : total_spent + money_left = 55 := by
  sorry

end NUMINAMATH_CALUDE_mom_initial_money_l2155_215546


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2155_215565

theorem sum_of_reciprocals_of_roots (p q : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + p*x₁ + q = 0 → 
  x₂^2 + p*x₂ + q = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  1/x₁ + 1/x₂ = -p/q :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2155_215565


namespace NUMINAMATH_CALUDE_greatest_integer_y_l2155_215536

theorem greatest_integer_y (y : ℕ+) : (y.val : ℝ)^4 / (y.val : ℝ)^2 < 18 ↔ y.val ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_y_l2155_215536


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2155_215585

/-- The surface area of a sphere circumscribing a rectangular solid -/
theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (S : ℝ) :
  a = 3 →
  b = 4 →
  c = 5 →
  S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
  S = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2155_215585


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2155_215508

/-- A cubic function with specific properties -/
structure CubicFunction where
  b : ℝ
  c : ℝ
  d : ℝ
  f : ℝ → ℝ
  f_def : ∀ x, f x = x^3 + 3*b*x^2 + c*x + d
  increasing_neg : ∀ x y, x < y → y < 0 → f x < f y
  decreasing_pos : ∀ x y, 0 < x → x < y → y < 2 → f y < f x
  root_neg_b : f (-b) = 0

/-- Main theorem about the cubic function -/
theorem cubic_function_properties (cf : CubicFunction) :
  cf.c = 0 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ -cf.b ∧ x₂ ≠ -cf.b ∧ cf.f x₁ = 0 ∧ cf.f x₂ = 0 ∧ x₂ - (-cf.b) = (-cf.b) - x₁) ∧
  (0 ≤ cf.f 1 ∧ cf.f 1 < 11) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2155_215508


namespace NUMINAMATH_CALUDE_black_marble_price_is_ten_cents_l2155_215535

/-- Represents the marble pricing problem --/
structure MarbleProblem where
  total_marbles : ℕ
  white_percentage : ℚ
  black_percentage : ℚ
  white_price : ℚ
  color_price : ℚ
  total_earnings : ℚ

/-- Calculates the price of each black marble --/
def black_marble_price (p : MarbleProblem) : ℚ :=
  let white_marbles := p.white_percentage * p.total_marbles
  let black_marbles := p.black_percentage * p.total_marbles
  let color_marbles := p.total_marbles - (white_marbles + black_marbles)
  let white_earnings := white_marbles * p.white_price
  let color_earnings := color_marbles * p.color_price
  let black_earnings := p.total_earnings - (white_earnings + color_earnings)
  black_earnings / black_marbles

/-- Theorem stating that the black marble price is $0.10 --/
theorem black_marble_price_is_ten_cents 
  (p : MarbleProblem) 
  (h1 : p.total_marbles = 100)
  (h2 : p.white_percentage = 1/5)
  (h3 : p.black_percentage = 3/10)
  (h4 : p.white_price = 1/20)
  (h5 : p.color_price = 1/5)
  (h6 : p.total_earnings = 14) :
  black_marble_price p = 1/10 := by
  sorry

#eval black_marble_price { 
  total_marbles := 100, 
  white_percentage := 1/5, 
  black_percentage := 3/10, 
  white_price := 1/5, 
  color_price := 1/5, 
  total_earnings := 14 
}

end NUMINAMATH_CALUDE_black_marble_price_is_ten_cents_l2155_215535


namespace NUMINAMATH_CALUDE_odd_function_sum_l2155_215586

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_even : is_even (λ x => f (x + 2))) 
  (h_f1 : f 1 = 1) : 
  f 8 + f 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2155_215586


namespace NUMINAMATH_CALUDE_prob_all_even_four_dice_l2155_215531

/-- The probability of a single standard six-sided die showing an even number -/
def prob_even_single : ℚ := 1 / 2

/-- The number of dice being tossed simultaneously -/
def num_dice : ℕ := 4

/-- Theorem: The probability of all four standard six-sided dice showing even numbers
    when tossed simultaneously is 1/16 -/
theorem prob_all_even_four_dice :
  (prob_even_single ^ num_dice : ℚ) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_even_four_dice_l2155_215531


namespace NUMINAMATH_CALUDE_candy_game_theorem_l2155_215591

/-- The maximum number of candies that can be eaten in the candy-eating game. -/
def max_candies (n : ℕ) : ℕ :=
  n.choose 2

/-- The candy-eating game theorem. -/
theorem candy_game_theorem :
  max_candies 27 = 351 :=
by sorry

end NUMINAMATH_CALUDE_candy_game_theorem_l2155_215591


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l2155_215512

theorem ceiling_floor_product (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 132 → -12 < y ∧ y < -11 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l2155_215512


namespace NUMINAMATH_CALUDE_g_inverse_composition_l2155_215549

def g : Fin 5 → Fin 5
| 0 => 3  -- Representing g(1) = 4
| 1 => 2  -- Representing g(2) = 3
| 2 => 0  -- Representing g(3) = 1
| 3 => 4  -- Representing g(4) = 5
| 4 => 1  -- Representing g(5) = 2

theorem g_inverse_composition (h : Function.Bijective g) :
  (Function.invFun g) ((Function.invFun g) ((Function.invFun g) 2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_composition_l2155_215549


namespace NUMINAMATH_CALUDE_sum_of_z_values_l2155_215525

-- Define the function f
def f (x : ℝ) : ℝ := (4*x)^2 - 3*(4*x) + 2

-- State the theorem
theorem sum_of_z_values (f : ℝ → ℝ) : 
  (f = λ x => (4*x)^2 - 3*(4*x) + 2) → 
  (∃ z₁ z₂ : ℝ, f z₁ = 9 ∧ f z₂ = 9 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 3/16) := by
  sorry


end NUMINAMATH_CALUDE_sum_of_z_values_l2155_215525


namespace NUMINAMATH_CALUDE_ellipse_with_given_properties_l2155_215545

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity

/-- The equation of an ellipse in standard form -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

theorem ellipse_with_given_properties :
  ∀ (E : Ellipse),
    E.b = 1 →  -- Half of minor axis length is 1
    E.e = Real.sqrt 2 / 2 →  -- Eccentricity is √2/2
    (∀ x y : ℝ, ellipse_equation E x y ↔ x^2 / 2 + y^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_with_given_properties_l2155_215545


namespace NUMINAMATH_CALUDE_problem_solution_l2155_215558

theorem problem_solution (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : 
  |x - y| = 10 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2155_215558


namespace NUMINAMATH_CALUDE_point_q_coordinates_l2155_215529

/-- Given two points P and Q in a 2D Cartesian coordinate system,
    prove that Q has coordinates (1, -3) under the given conditions. -/
theorem point_q_coordinates
  (P Q : ℝ × ℝ)  -- P and Q are points in 2D space
  (h1 : P = (1, 2))  -- P has coordinates (1, 2)
  (h2 : (Q.2 : ℝ) < 0)  -- Q is below the x-axis
  (h3 : P.1 = Q.1)  -- PQ is parallel to the y-axis
  (h4 : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 5)  -- PQ = 5
  : Q = (1, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_q_coordinates_l2155_215529


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2155_215562

/-- Two lines are parallel if and only if their slopes are equal but they are not identical --/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ / n₁ = m₂ / n₂ ∧ m₁ / c₁ ≠ m₂ / c₂

/-- The problem statement --/
theorem parallel_lines_a_value :
  ∀ a : ℝ,
  are_parallel 1 a 6 (a - 2) 3 (2 * a) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2155_215562


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2155_215521

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 1 + a 3 + a 5 = 21 →
  a 2 + a 4 + a 6 = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2155_215521


namespace NUMINAMATH_CALUDE_cosine_domain_range_minimum_l2155_215505

open Real

theorem cosine_domain_range_minimum (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x = cos x) →
  (∀ x ∈ Set.Icc a b, -1/2 ≤ f x ∧ f x ≤ 1) →
  (∃ x ∈ Set.Icc a b, f x = -1/2) →
  (∃ x ∈ Set.Icc a b, f x = 1) →
  b - a ≥ 2*π/3 :=
by sorry

end NUMINAMATH_CALUDE_cosine_domain_range_minimum_l2155_215505


namespace NUMINAMATH_CALUDE_cookie_distribution_l2155_215567

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) :
  total_cookies = 24 →
  num_people = 6 →
  total_cookies = num_people * cookies_per_person →
  cookies_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l2155_215567


namespace NUMINAMATH_CALUDE_davids_physics_marks_l2155_215571

def english_marks : ℕ := 45
def math_marks : ℕ := 35
def chemistry_marks : ℕ := 47
def biology_marks : ℕ := 55
def average_marks : ℚ := 46.8
def num_subjects : ℕ := 5

theorem davids_physics_marks :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    physics_marks = 52 := by
  sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l2155_215571


namespace NUMINAMATH_CALUDE_jeans_prices_l2155_215590

/-- Represents the shopping scenario with Mary and her children --/
structure ShoppingScenario where
  coat_original_price : ℝ
  coat_discount_rate : ℝ
  backpack_cost : ℝ
  shoes_cost : ℝ
  subtotal : ℝ
  jeans_price_difference : ℝ
  sales_tax_rate : ℝ

/-- Theorem stating the prices of Jamie's jeans --/
theorem jeans_prices (scenario : ShoppingScenario)
  (h_coat : scenario.coat_original_price = 50)
  (h_discount : scenario.coat_discount_rate = 0.1)
  (h_backpack : scenario.backpack_cost = 25)
  (h_shoes : scenario.shoes_cost = 30)
  (h_subtotal : scenario.subtotal = 139)
  (h_difference : scenario.jeans_price_difference = 15)
  (h_tax : scenario.sales_tax_rate = 0.07) :
  ∃ (cheap_jeans expensive_jeans : ℝ),
    cheap_jeans = 12 ∧
    expensive_jeans = 27 ∧
    cheap_jeans + expensive_jeans = scenario.subtotal -
      (scenario.coat_original_price * (1 - scenario.coat_discount_rate) +
       scenario.backpack_cost + scenario.shoes_cost) ∧
    expensive_jeans - cheap_jeans = scenario.jeans_price_difference :=
by sorry

end NUMINAMATH_CALUDE_jeans_prices_l2155_215590


namespace NUMINAMATH_CALUDE_sunflower_height_comparison_l2155_215542

/-- Given that sunflowers from Packet A are 20% taller than those from Packet B,
    and sunflowers from Packet A are 192 inches tall,
    prove that sunflowers from Packet B are 160 inches tall. -/
theorem sunflower_height_comparison (height_A height_B : ℝ) : 
  height_A = height_B * 1.2 → height_A = 192 → height_B = 160 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_comparison_l2155_215542


namespace NUMINAMATH_CALUDE_g_properties_l2155_215500

noncomputable def g (x : ℝ) : ℝ := (4 * Real.sin x ^ 4 + 7 * Real.cos x ^ 2) / (4 * Real.cos x ^ 4 + Real.sin x ^ 2)

theorem g_properties :
  (∀ k : ℤ, g (Real.pi / 3 + k * Real.pi) = 4 ∧ g (-Real.pi / 3 + k * Real.pi) = 4 ∧ g (Real.pi / 2 + k * Real.pi) = 4) ∧
  (∀ x : ℝ, g x ≥ 7 / 4) ∧
  (∀ x : ℝ, g x ≤ 63 / 15) ∧
  (∃ x : ℝ, g x = 7 / 4) ∧
  (∃ x : ℝ, g x = 63 / 15) := by
  sorry

end NUMINAMATH_CALUDE_g_properties_l2155_215500


namespace NUMINAMATH_CALUDE_smallest_a_for_two_roots_less_than_one_l2155_215583

theorem smallest_a_for_two_roots_less_than_one : 
  ∃ (a b c : ℤ), 
    (a > 0) ∧ 
    (∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 0 < r₁ ∧ r₁ < 1 ∧ 0 < r₂ ∧ r₂ < 1 ∧ 
      (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = r₁ ∨ x = r₂)) ∧
    (∀ a' : ℤ, 0 < a' ∧ a' < a → 
      ¬∃ (b' c' : ℤ), ∃ (s₁ s₂ : ℝ), s₁ ≠ s₂ ∧ 0 < s₁ ∧ s₁ < 1 ∧ 0 < s₂ ∧ s₂ < 1 ∧ 
        (∀ x : ℝ, a' * x^2 + b' * x + c' = 0 ↔ x = s₁ ∨ x = s₂)) ∧
    a = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_two_roots_less_than_one_l2155_215583


namespace NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_A_l2155_215541

/-- Proves that the percentage of ethanol in fuel A is 12% -/
theorem ethanol_percentage_in_fuel_A :
  let tank_capacity : ℝ := 208
  let fuel_A_volume : ℝ := 82
  let fuel_B_ethanol_percentage : ℝ := 0.16
  let total_ethanol : ℝ := 30
  let fuel_B_volume : ℝ := tank_capacity - fuel_A_volume
  let fuel_A_ethanol_percentage : ℝ := (total_ethanol - fuel_B_ethanol_percentage * fuel_B_volume) / fuel_A_volume
  fuel_A_ethanol_percentage = 0.12 := by sorry

end NUMINAMATH_CALUDE_ethanol_percentage_in_fuel_A_l2155_215541


namespace NUMINAMATH_CALUDE_age_difference_of_children_l2155_215587

/-- Proves that the age difference between children is 4 years given the conditions -/
theorem age_difference_of_children (n : ℕ) (sum_ages : ℕ) (eldest_age : ℕ) (d : ℕ) :
  n = 4 ∧ 
  sum_ages = 48 ∧ 
  eldest_age = 18 ∧ 
  sum_ages = n * eldest_age - (d * (n * (n - 1)) / 2) →
  d = 4 :=
by sorry


end NUMINAMATH_CALUDE_age_difference_of_children_l2155_215587


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2155_215514

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let E := {(x, y) : ℝ × ℝ | x^2/a^2 - y^2/b^2 = 1}
  let A := (a, 0)
  let C := {(x, y) : ℝ × ℝ | y^2 = 8*a*x}
  let F := (2*a, 0)
  let asymptote := {(x, y) : ℝ × ℝ | y = b/a * x ∨ y = -b/a * x}
  ∃ P ∈ asymptote, (A.1 - P.1) * (F.1 - P.1) + (A.2 - P.2) * (F.2 - P.2) = 0 →
  let e := Real.sqrt (1 + b^2/a^2)
  1 < e ∧ e ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2155_215514


namespace NUMINAMATH_CALUDE_range_of_a_l2155_215506

-- Define the conditions
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x a : ℝ) : Prop := x < a

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x)

-- Theorem statement
theorem range_of_a (h : sufficient_not_necessary p (q · a)) :
  ∀ y : ℝ, y ≥ 2 ↔ ∃ x : ℝ, a = y := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2155_215506


namespace NUMINAMATH_CALUDE_union_equals_A_l2155_215568

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {y | m*y + 2 = 0}

theorem union_equals_A : {m : ℝ | A ∪ B m = A} = {0, -1, -2/3} := by sorry

end NUMINAMATH_CALUDE_union_equals_A_l2155_215568


namespace NUMINAMATH_CALUDE_fraction_problem_l2155_215530

theorem fraction_problem : ∃ x : ℚ, x < 20 / 100 * 180 ∧ x * 180 = 24 := by
  use 2 / 15
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2155_215530


namespace NUMINAMATH_CALUDE_map_scale_conversion_l2155_215598

/-- Given a map scale where 10 cm represents 50 km, prove that 15 cm represents 75 km -/
theorem map_scale_conversion (scale : ℝ) (h1 : scale = 50 / 10) : 15 * scale = 75 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l2155_215598


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2155_215555

/-- For a hyperbola with equation x²/a² - y²/b² = 1, if the distance between
    its vertices (2a) is one-third of its focal length (2c), then its
    eccentricity (e) is equal to 3. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) :
  (2 * a = (1/3) * (2 * c)) → (c / a = 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2155_215555


namespace NUMINAMATH_CALUDE_division_problem_l2155_215593

theorem division_problem (A : ℕ) (h1 : 59 / A = 6) (h2 : 59 % A = 5) : A = 9 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2155_215593


namespace NUMINAMATH_CALUDE_longest_side_is_72_l2155_215540

def rectangle_problem (length width : ℝ) : Prop :=
  length > 0 ∧ 
  width > 0 ∧ 
  2 * (length + width) = 240 ∧ 
  length * width = 12 * 240

theorem longest_side_is_72 : 
  ∃ (length width : ℝ), 
    rectangle_problem length width ∧ 
    (length ≥ width → length = 72) ∧
    (width > length → width = 72) :=
sorry

end NUMINAMATH_CALUDE_longest_side_is_72_l2155_215540


namespace NUMINAMATH_CALUDE_angle_B_value_min_side_b_value_l2155_215563

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def TriangleConditions (t : Triangle) : Prop :=
  (Real.cos t.C / Real.cos t.B = (2 * t.a - t.c) / t.b) ∧
  (t.a + t.c = 2)

theorem angle_B_value (t : Triangle) (h : TriangleConditions t) : t.B = π / 3 := by
  sorry

theorem min_side_b_value (t : Triangle) (h : TriangleConditions t) : 
  ∃ (b_min : ℝ), b_min = 1 ∧ ∀ (t' : Triangle), TriangleConditions t' → t'.b ≥ b_min := by
  sorry

end NUMINAMATH_CALUDE_angle_B_value_min_side_b_value_l2155_215563


namespace NUMINAMATH_CALUDE_santiago_garrett_rose_difference_l2155_215550

/-- Mrs. Santiago has 58 red roses and Mrs. Garrett has 24 red roses. 
    The theorem proves that Mrs. Santiago has 34 more red roses than Mrs. Garrett. -/
theorem santiago_garrett_rose_difference :
  ∀ (santiago_roses garrett_roses : ℕ),
    santiago_roses = 58 →
    garrett_roses = 24 →
    santiago_roses - garrett_roses = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_santiago_garrett_rose_difference_l2155_215550


namespace NUMINAMATH_CALUDE_exists_crocodile_coloring_l2155_215578

/-- A coloring function for the infinite chess grid -/
def GridColoring := ℤ → ℤ → Fin 2

/-- The crocodile move property for a given coloring -/
def IsCrocodileColoring (f : GridColoring) (m n : ℕ+) : Prop :=
  ∀ x y : ℤ, f x y ≠ f (x + m) (y + n) ∧ f x y ≠ f (x + n) (y + m)

/-- Theorem: For any positive integers m and n, there exists a valid crocodile coloring -/
theorem exists_crocodile_coloring (m n : ℕ+) :
  ∃ f : GridColoring, IsCrocodileColoring f m n := by
  sorry

end NUMINAMATH_CALUDE_exists_crocodile_coloring_l2155_215578


namespace NUMINAMATH_CALUDE_nancys_weight_l2155_215572

theorem nancys_weight (water_intake : ℝ) (water_percentage : ℝ) :
  water_intake = 54 →
  water_percentage = 0.60 →
  water_intake = water_percentage * 90 :=
by
  sorry

end NUMINAMATH_CALUDE_nancys_weight_l2155_215572


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2155_215597

/-- Calculates the total shaded area of a right triangle and half of an adjacent rectangle -/
theorem shaded_area_calculation (triangle_base : ℝ) (triangle_height : ℝ) (rectangle_width : ℝ) :
  triangle_base = 6 →
  triangle_height = 8 →
  rectangle_width = 5 →
  (1 / 2 * triangle_base * triangle_height) + (1 / 2 * rectangle_width * triangle_height) = 44 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2155_215597


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l2155_215596

/-- Calculates the gain percent given the cost price and selling price -/
def gainPercent (costPrice sellingPrice : ℚ) : ℚ :=
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Proves that the gain percent on a cycle bought for Rs. 930 and sold for Rs. 1210 is approximately 30.11% -/
theorem cycle_gain_percent :
  let costPrice : ℚ := 930
  let sellingPrice : ℚ := 1210
  abs (gainPercent costPrice sellingPrice - 30.11) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l2155_215596


namespace NUMINAMATH_CALUDE_exists_multiple_without_zero_l2155_215544

def has_no_zero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

theorem exists_multiple_without_zero (k : ℕ) : 
  k > 0 → ∃ n : ℕ, 5^k ∣ n ∧ has_no_zero n :=
sorry

end NUMINAMATH_CALUDE_exists_multiple_without_zero_l2155_215544


namespace NUMINAMATH_CALUDE_coinciding_rest_days_main_theorem_l2155_215543

/-- Chris's schedule cycle length -/
def chris_cycle : ℕ := 7

/-- Dana's schedule cycle length -/
def dana_cycle : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 500

/-- Number of rest days for Chris in one cycle -/
def chris_rest_days : ℕ := 2

/-- Number of rest days for Dana in one cycle -/
def dana_rest_days : ℕ := 1

/-- The number of days both Chris and Dana have rest-days on the same day
    within the first 500 days of their schedules -/
theorem coinciding_rest_days : ℕ := by
  sorry

/-- The main theorem stating that the number of coinciding rest days is 28 -/
theorem main_theorem : coinciding_rest_days = 28 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_main_theorem_l2155_215543


namespace NUMINAMATH_CALUDE_ratio_equality_l2155_215551

theorem ratio_equality (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) : (a / 5) / (b / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2155_215551


namespace NUMINAMATH_CALUDE_divisibility_rules_l2155_215589

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ := n % 10

-- Define a function to get the last two digits of a natural number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define a function to check if a number is even
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define a function to sum the digits of a natural number
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

theorem divisibility_rules (n : ℕ) :
  (n % 2 = 0 ↔ isEven (lastDigit n)) ∧
  (n % 5 = 0 ↔ lastDigit n = 0 ∨ lastDigit n = 5) ∧
  (n % 3 = 0 ↔ sumOfDigits n % 3 = 0) ∧
  (n % 4 = 0 ↔ lastTwoDigits n % 4 = 0) ∧
  (n % 25 = 0 ↔ lastTwoDigits n % 25 = 0) :=
by sorry


end NUMINAMATH_CALUDE_divisibility_rules_l2155_215589


namespace NUMINAMATH_CALUDE_largest_angle_is_112_5_l2155_215560

/-- Represents a quadrilateral formed by folding two sides of a square along its diagonal -/
structure FoldedSquare where
  /-- The side length of the original square -/
  side : ℝ
  /-- Assumption that the side length is positive -/
  side_pos : side > 0

/-- The largest angle in the folded square -/
def largest_angle (fs : FoldedSquare) : ℝ := 112.5

/-- Theorem stating that the largest angle in the folded square is 112.5° -/
theorem largest_angle_is_112_5 (fs : FoldedSquare) :
  largest_angle fs = 112.5 := by sorry

end NUMINAMATH_CALUDE_largest_angle_is_112_5_l2155_215560


namespace NUMINAMATH_CALUDE_f_has_two_real_roots_l2155_215507

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 3

/-- Theorem stating that f has exactly two real roots -/
theorem f_has_two_real_roots : ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_two_real_roots_l2155_215507


namespace NUMINAMATH_CALUDE_smallest_n_value_l2155_215502

/-- Represents a rectangular block made of 1-cm cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in the block -/
def Block.totalCubes (b : Block) : ℕ := b.length * b.width * b.height

/-- Calculates the number of invisible cubes when three faces are visible -/
def Block.invisibleCubes (b : Block) : ℕ := (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- Theorem stating the smallest possible value of N -/
theorem smallest_n_value (b : Block) (h : b.invisibleCubes = 300) :
  ∃ (min_b : Block), min_b.invisibleCubes = 300 ∧
    min_b.totalCubes ≤ b.totalCubes ∧
    min_b.totalCubes = 468 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l2155_215502


namespace NUMINAMATH_CALUDE_remainder_property_l2155_215554

theorem remainder_property (n : ℕ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_property_l2155_215554


namespace NUMINAMATH_CALUDE_big_suv_to_normal_car_ratio_l2155_215524

/-- Represents the time in minutes for each task when washing a normal car -/
structure NormalCarWashTime where
  windows : Nat
  body : Nat
  tires : Nat
  waxing : Nat

/-- Calculates the total time to wash a normal car -/
def normalCarTotalTime (t : NormalCarWashTime) : Nat :=
  t.windows + t.body + t.tires + t.waxing

/-- Represents the washing scenario -/
structure CarWashScenario where
  normalCarTime : NormalCarWashTime
  normalCarCount : Nat
  totalTime : Nat

/-- Theorem: The ratio of time taken to wash the big SUV to the time taken to wash a normal car is 2:1 -/
theorem big_suv_to_normal_car_ratio 
  (scenario : CarWashScenario) 
  (h1 : scenario.normalCarTime = ⟨4, 7, 4, 9⟩) 
  (h2 : scenario.normalCarCount = 2) 
  (h3 : scenario.totalTime = 96) : 
  (scenario.totalTime - scenario.normalCarCount * normalCarTotalTime scenario.normalCarTime) / 
  (normalCarTotalTime scenario.normalCarTime) = 2 := by
  sorry


end NUMINAMATH_CALUDE_big_suv_to_normal_car_ratio_l2155_215524


namespace NUMINAMATH_CALUDE_sum_of_squares_l2155_215520

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 90)
  (h2 : x^2 * y + x * y^2 = 1122) : 
  x^2 + y^2 = 1044 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2155_215520


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l2155_215569

theorem quadratic_form_ratio (x : ℝ) : 
  ∃ (d e : ℝ), x^2 + 800*x + 500 = (x + d)^2 + e ∧ e / d = -398.75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l2155_215569


namespace NUMINAMATH_CALUDE_shortest_distance_curve_to_line_l2155_215581

/-- The shortest distance from a point on the curve y = 2ln x to the line 2x - y + 3 = 0 is √5 -/
theorem shortest_distance_curve_to_line :
  let curve := fun x : ℝ => 2 * Real.log x
  let line := fun x y : ℝ => 2 * x - y + 3 = 0
  ∃ d : ℝ, d = Real.sqrt 5 ∧
    ∀ x y : ℝ, curve x = y →
      d ≤ (|2 * x - y + 3| / Real.sqrt (2^2 + (-1)^2)) ∧
      ∃ x₀ y₀ : ℝ, curve x₀ = y₀ ∧
        d = (|2 * x₀ - y₀ + 3| / Real.sqrt (2^2 + (-1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_curve_to_line_l2155_215581


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2155_215553

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧ 
  (∀ x : ℝ, x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2155_215553


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_50_l2155_215594

theorem least_product_of_primes_above_50 (p q : ℕ) : 
  p.Prime → q.Prime → p > 50 → q > 50 → p ≠ q → 
  ∀ r s : ℕ, r.Prime → s.Prime → r > 50 → s > 50 → r ≠ s → 
  p * q ≤ r * s :=
sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_50_l2155_215594


namespace NUMINAMATH_CALUDE_ball_trajectory_l2155_215539

-- Define the quadratic function
def f (t : ℚ) : ℚ := -4.9 * t^2 + 7 * t + 10

-- State the theorem
theorem ball_trajectory :
  f (5/7) = 15 ∧
  f (10/7) = 0 ∧
  ∀ t : ℚ, 5/7 < t → t < 10/7 → f t ≠ 15 ∧ f t ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ball_trajectory_l2155_215539


namespace NUMINAMATH_CALUDE_scientific_notation_of_12417_l2155_215580

theorem scientific_notation_of_12417 : ∃ (a : ℝ) (n : ℤ), 
  12417 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.2417 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_12417_l2155_215580


namespace NUMINAMATH_CALUDE_quadratic_no_solution_l2155_215582

theorem quadratic_no_solution : 
  {x : ℝ | x^2 - 2*x + 3 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_l2155_215582


namespace NUMINAMATH_CALUDE_coffee_maker_price_l2155_215503

theorem coffee_maker_price (sale_price : ℝ) (discount : ℝ) (original_price : ℝ) : 
  sale_price = 70 → discount = 20 → original_price = sale_price + discount → original_price = 90 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_price_l2155_215503


namespace NUMINAMATH_CALUDE_chess_tournament_matches_l2155_215533

/-- Represents a single elimination tournament --/
structure Tournament :=
  (total_players : ℕ)
  (bye_players : ℕ)
  (h_bye : bye_players < total_players)

/-- Calculates the number of matches in a tournament --/
def matches_played (t : Tournament) : ℕ := t.total_players - 1

/-- Main theorem about the chess tournament --/
theorem chess_tournament_matches :
  ∃ (t : Tournament),
    t.total_players = 120 ∧
    t.bye_players = 40 ∧
    matches_played t = 119 ∧
    119 % 7 = 0 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_matches_l2155_215533


namespace NUMINAMATH_CALUDE_max_sum_rectangle_sides_l2155_215548

theorem max_sum_rectangle_sides (n : ℕ) (h : n = 10) :
  let total_sum := n * (n + 1) / 2
  let corner_sum := n + (n - 1) + (n - 2) + (n - 4)
  ∃ (side_sum : ℕ), 
    side_sum = (total_sum + corner_sum) / 4 ∧ 
    side_sum = 22 ∧
    ∀ (other_sum : ℕ), 
      (other_sum * 4 ≤ total_sum + corner_sum) → 
      other_sum ≤ side_sum :=
by sorry

end NUMINAMATH_CALUDE_max_sum_rectangle_sides_l2155_215548


namespace NUMINAMATH_CALUDE_paul_homework_hours_l2155_215573

/-- Calculates the total hours of homework on weeknights for Paul --/
def weeknight_homework (total_weeknights : ℕ) (practice_nights : ℕ) (average_hours : ℕ) : ℕ :=
  (total_weeknights - practice_nights) * average_hours

/-- Proves that Paul has 9 hours of homework on weeknights --/
theorem paul_homework_hours :
  let total_weeknights := 5
  let practice_nights := 2
  let average_hours := 3
  weeknight_homework total_weeknights practice_nights average_hours = 9 := by
  sorry

#eval weeknight_homework 5 2 3

end NUMINAMATH_CALUDE_paul_homework_hours_l2155_215573


namespace NUMINAMATH_CALUDE_max_value_of_function_l2155_215501

theorem max_value_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 9 ≤ 6) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 9 = 6) →
  a = 3 ∨ a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2155_215501


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_one_l2155_215592

theorem x_plus_y_equals_negative_one (x y : ℝ) : 
  (x - 1)^2 + |y + 2| = 0 → x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_one_l2155_215592


namespace NUMINAMATH_CALUDE_arctan_equality_l2155_215515

theorem arctan_equality : 4 * Real.arctan (1/5) - Real.arctan (1/239) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equality_l2155_215515


namespace NUMINAMATH_CALUDE_soccer_enjoyment_misreporting_l2155_215523

theorem soccer_enjoyment_misreporting (total : ℝ) (total_pos : 0 < total) :
  let enjoy := 0.7 * total
  let dont_enjoy := 0.3 * total
  let say_dont_but_do := 0.25 * enjoy
  let say_dont_and_dont := 0.85 * dont_enjoy
  say_dont_but_do / (say_dont_but_do + say_dont_and_dont) = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_soccer_enjoyment_misreporting_l2155_215523


namespace NUMINAMATH_CALUDE_overall_average_marks_l2155_215584

/-- Given three batches of students with their respective sizes and average marks,
    calculate the overall average marks for all students combined. -/
theorem overall_average_marks
  (batch1_size batch2_size batch3_size : ℕ)
  (batch1_avg batch2_avg batch3_avg : ℚ)
  (h1 : batch1_size = 40)
  (h2 : batch2_size = 50)
  (h3 : batch3_size = 60)
  (h4 : batch1_avg = 45)
  (h5 : batch2_avg = 55)
  (h6 : batch3_avg = 65) :
  (batch1_size * batch1_avg + batch2_size * batch2_avg + batch3_size * batch3_avg) /
  (batch1_size + batch2_size + batch3_size) = 8450 / 150 := by
  sorry

#eval (8450 : ℚ) / 150  -- To verify the result

end NUMINAMATH_CALUDE_overall_average_marks_l2155_215584


namespace NUMINAMATH_CALUDE_rectangle_area_l2155_215519

theorem rectangle_area (w : ℝ) (h₁ : w > 0) (h₂ : 10 * w = 200) : w * (4 * w) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2155_215519


namespace NUMINAMATH_CALUDE_min_abs_E_is_zero_l2155_215528

/-- Given a real-valued function E, prove that its minimum absolute value is 0
    when the minimum of |E(x)| + |x + 6| + |x - 5| is 11 for all real x. -/
theorem min_abs_E_is_zero (E : ℝ → ℝ) : 
  (∀ x, |E x| + |x + 6| + |x - 5| ≥ 11) → 
  (∃ x, |E x| + |x + 6| + |x - 5| = 11) → 
  ∃ x, |E x| = 0 :=
sorry

end NUMINAMATH_CALUDE_min_abs_E_is_zero_l2155_215528


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_five_l2155_215527

theorem complex_expression_equals_negative_five :
  Real.sqrt 27 + (-1/3)⁻¹ - |2 - Real.sqrt 3| - 8 * Real.cos (30 * π / 180) = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_five_l2155_215527


namespace NUMINAMATH_CALUDE_valid_selections_count_l2155_215518

def male_teachers : ℕ := 5
def female_teachers : ℕ := 4
def total_teachers : ℕ := male_teachers + female_teachers
def selected_teachers : ℕ := 3

def all_selections : ℕ := (total_teachers.choose selected_teachers)
def all_male_selections : ℕ := (male_teachers.choose selected_teachers)
def all_female_selections : ℕ := (female_teachers.choose selected_teachers)

theorem valid_selections_count : 
  all_selections - (all_male_selections + all_female_selections) = 420 := by
  sorry

end NUMINAMATH_CALUDE_valid_selections_count_l2155_215518
