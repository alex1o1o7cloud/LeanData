import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2888_288892

/-- 
Given a rectangular prism with length l, width w, and height h,
where l = 2w, w = 2h, and the sum of all edge lengths is 56,
prove that the volume is 64.
-/
theorem rectangular_prism_volume (l w h : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : w = 2 * h) 
  (h3 : 4 * (l + w + h) = 56) : 
  l * w * h = 64 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2888_288892


namespace NUMINAMATH_CALUDE_two_balls_in_five_boxes_with_adjacent_empty_l2888_288840

/-- Represents the number of ways to arrange two balls in two boxes -/
def A_2_2 : ℕ := 2

/-- Represents the number of distinct boxes -/
def num_boxes : ℕ := 5

/-- Represents the number of balls -/
def num_balls : ℕ := 2

/-- Represents the number of empty boxes -/
def num_empty_boxes : ℕ := num_boxes - num_balls

/-- Represents the number of adjacent empty box pairs -/
def num_adjacent_empty_pairs : ℕ := 4

/-- The main theorem to prove -/
theorem two_balls_in_five_boxes_with_adjacent_empty : 
  (2 * A_2_2 + A_2_2 + A_2_2 + 2 * A_2_2 : ℕ) = 12 := by
  sorry


end NUMINAMATH_CALUDE_two_balls_in_five_boxes_with_adjacent_empty_l2888_288840


namespace NUMINAMATH_CALUDE_factorization_of_ax2_minus_16ay2_l2888_288870

theorem factorization_of_ax2_minus_16ay2 (a x y : ℝ) : 
  a * x^2 - 16 * a * y^2 = a * (x + 4*y) * (x - 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_ax2_minus_16ay2_l2888_288870


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2888_288843

theorem quadratic_equation_solution :
  let x₁ : ℝ := 1 + Real.sqrt 2 / 2
  let x₂ : ℝ := 1 - Real.sqrt 2 / 2
  (2 * x₁^2 - 4 * x₁ + 1 = 0) ∧ (2 * x₂^2 - 4 * x₂ + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2888_288843


namespace NUMINAMATH_CALUDE_arthur_wallet_problem_l2888_288863

theorem arthur_wallet_problem (initial_amount : ℝ) (spent_fraction : ℝ) (remaining_amount : ℝ) : 
  initial_amount = 200 →
  spent_fraction = 4/5 →
  remaining_amount = initial_amount - (spent_fraction * initial_amount) →
  remaining_amount = 40 := by
sorry

end NUMINAMATH_CALUDE_arthur_wallet_problem_l2888_288863


namespace NUMINAMATH_CALUDE_smallest_block_with_270_hidden_cubes_l2888_288872

/-- Represents the dimensions of a rectangular block --/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block --/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of hidden cubes in a block --/
def hiddenCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Theorem stating the smallest possible value of N --/
theorem smallest_block_with_270_hidden_cubes :
  ∃ (d : BlockDimensions),
    hiddenCubes d = 270 ∧
    (∀ (d' : BlockDimensions), hiddenCubes d' = 270 → totalCubes d ≤ totalCubes d') ∧
    totalCubes d = 420 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_with_270_hidden_cubes_l2888_288872


namespace NUMINAMATH_CALUDE_distance_one_fourth_from_perigee_l2888_288862

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ
  apogee : ℝ

/-- Calculates the distance from the focus to a point on the major axis of an elliptical orbit -/
def distanceFromFocus (orbit : EllipticalOrbit) (fraction : ℝ) : ℝ :=
  let majorAxis := orbit.apogee + orbit.perigee
  let centerToFocus := Real.sqrt ((majorAxis / 2) ^ 2 - orbit.perigee ^ 2)
  let distanceFromPerigee := fraction * majorAxis
  distanceFromPerigee

/-- Theorem: For an elliptical orbit with perigee 3 AU and apogee 15 AU,
    the distance from the focus to a point 1/4 of the way from perigee to apogee
    along the major axis is 4.5 AU -/
theorem distance_one_fourth_from_perigee (orbit : EllipticalOrbit)
    (h1 : orbit.perigee = 3)
    (h2 : orbit.apogee = 15) :
    distanceFromFocus orbit (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_one_fourth_from_perigee_l2888_288862


namespace NUMINAMATH_CALUDE_problem_statement_l2888_288807

/-- Given two natural numbers a and b, returns the floor of a/b -/
def floorDiv (a b : ℕ) : ℕ := a / b

/-- Returns true if the number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- Counts the number of prime numbers in the range (a, b) -/
def countPrimesBetween (a b : ℕ) : ℕ := sorry

theorem problem_statement :
  ∃ n : ℕ,
    n = floorDiv 51 13 ∧
    countPrimesBetween n (floorDiv 89 9) = 2 ∧
    n = 3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2888_288807


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2888_288896

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2888_288896


namespace NUMINAMATH_CALUDE_rectangle_length_is_16_l2888_288814

/-- Represents a rectangle with given perimeter and length-width relationship -/
structure Rectangle where
  perimeter : ℝ
  width : ℝ
  length : ℝ
  perimeter_eq : perimeter = 2 * (length + width)
  length_eq : length = 2 * width

/-- Theorem: For a rectangle with perimeter 48 and length twice the width, the length is 16 -/
theorem rectangle_length_is_16 (rect : Rectangle) 
  (h_perimeter : rect.perimeter = 48) : rect.length = 16 := by
  sorry

#check rectangle_length_is_16

end NUMINAMATH_CALUDE_rectangle_length_is_16_l2888_288814


namespace NUMINAMATH_CALUDE_half_plus_six_equals_eleven_l2888_288893

theorem half_plus_six_equals_eleven (n : ℝ) : (1/2) * n + 6 = 11 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_six_equals_eleven_l2888_288893


namespace NUMINAMATH_CALUDE_points_always_odd_l2888_288849

/-- The number of points after k operations of adding a point between every two neighboring points. -/
def num_points (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then n
  else 2 * (num_points n (k - 1)) - 1

/-- Theorem: The number of points is always odd after each operation. -/
theorem points_always_odd (n : ℕ) (k : ℕ) (h : n ≥ 2) :
  Odd (num_points n k) :=
sorry

end NUMINAMATH_CALUDE_points_always_odd_l2888_288849


namespace NUMINAMATH_CALUDE_irrational_floor_bijection_l2888_288888

theorem irrational_floor_bijection (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hirra : Irrational a) (hirrb : Irrational b) 
  (hab : 1 / a + 1 / b = 1) :
  ∀ k : ℕ, ∃! (m n : ℕ), (⌊m * a⌋ = k ∨ ⌊n * b⌋ = k) ∧
    ∀ (p q : ℕ), (⌊p * a⌋ = k ∨ ⌊q * b⌋ = k) → (p = m ∧ ⌊p * a⌋ = k) ∨ (q = n ∧ ⌊q * b⌋ = k) :=
by sorry

end NUMINAMATH_CALUDE_irrational_floor_bijection_l2888_288888


namespace NUMINAMATH_CALUDE_probability_three_odd_in_six_rolls_l2888_288884

/-- The probability of getting an odd number on a single roll of a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of rolls -/
def num_rolls : ℕ := 6

/-- The number of desired odd outcomes -/
def desired_odd : ℕ := 3

/-- The probability of getting exactly k successes in n trials 
    with probability p for each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem probability_three_odd_in_six_rolls :
  binomial_probability num_rolls desired_odd prob_odd = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_odd_in_six_rolls_l2888_288884


namespace NUMINAMATH_CALUDE_exterior_angle_bisector_theorem_l2888_288841

/-- Given a triangle ABC with interior angles α, β, γ, and a triangle formed by 
    the bisectors of its exterior angles with angles α₁, β₁, γ₁, prove that 
    α = 180° - 2α₁, β = 180° - 2β₁, and γ = 180° - 2γ₁ --/
theorem exterior_angle_bisector_theorem 
  (α β γ α₁ β₁ γ₁ : Real) 
  (h_triangle : α + β + γ = 180) 
  (h_exterior_bisector : α₁ + β₁ + γ₁ = 180) : 
  α = 180 - 2*α₁ ∧ β = 180 - 2*β₁ ∧ γ = 180 - 2*γ₁ := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_bisector_theorem_l2888_288841


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2888_288865

-- Define the equation y^2 = 4
def equation (y : ℝ) : Prop := y^2 = 4

-- Define the statement y = 2
def statement (y : ℝ) : Prop := y = 2

-- Theorem: y = 2 is a sufficient but not necessary condition for y^2 = 4
theorem sufficient_but_not_necessary :
  (∀ y : ℝ, statement y → equation y) ∧
  ¬(∀ y : ℝ, equation y → statement y) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2888_288865


namespace NUMINAMATH_CALUDE_jasons_football_games_l2888_288886

theorem jasons_football_games (this_month next_month total : ℕ) 
  (h1 : this_month = 11)
  (h2 : next_month = 16)
  (h3 : total = 44) :
  ∃ last_month : ℕ, last_month + this_month + next_month = total ∧ last_month = 17 := by
  sorry

end NUMINAMATH_CALUDE_jasons_football_games_l2888_288886


namespace NUMINAMATH_CALUDE_x_range_given_inequality_l2888_288836

theorem x_range_given_inequality :
  (∀ t : ℝ, -1 ≤ t ∧ t ≤ 3 → 
    ∀ x : ℝ, x^2 - (t^2 + t - 3)*x + t^2*(t - 3) > 0) →
  ∀ x : ℝ, x ∈ (Set.Iio (-4) ∪ Set.Ioi 9) :=
by sorry

end NUMINAMATH_CALUDE_x_range_given_inequality_l2888_288836


namespace NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l2888_288897

/-- A police emergency number is a positive integer that ends with 133 in decimal notation. -/
def is_police_emergency_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 1000 * k + 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_has_large_prime_divisor (n : ℕ) 
  (h : is_police_emergency_number n) : 
  ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := by
sorry

end NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l2888_288897


namespace NUMINAMATH_CALUDE_journey_duration_l2888_288811

/-- Represents Tom's journey to Virgo island -/
def TomJourney : Prop :=
  let first_flight : ℝ := 5
  let first_layover : ℝ := 1
  let second_flight : ℝ := 2 * first_flight
  let second_layover : ℝ := 2
  let third_flight : ℝ := first_flight / 2
  let third_layover : ℝ := 3
  let first_boat : ℝ := 1.5
  let final_layover : ℝ := 0.75
  let final_boat : ℝ := 2 * (first_flight - third_flight)
  let total_time : ℝ := first_flight + first_layover + second_flight + second_layover +
                        third_flight + third_layover + first_boat + final_layover + final_boat
  total_time = 30.75

theorem journey_duration : TomJourney := by
  sorry

end NUMINAMATH_CALUDE_journey_duration_l2888_288811


namespace NUMINAMATH_CALUDE_base_is_six_l2888_288895

/-- The sum of all single-digit numbers in base b, including 0 and twice the largest single-digit number -/
def sum_digits (b : ℕ) : ℚ :=
  (b^2 + 3*b - 4) / 2

/-- Representation of 107 in base b -/
def base_b_107 (b : ℕ) : ℕ :=
  b^2 + 7

theorem base_is_six (b : ℕ) (h_pos : b > 0) :
  sum_digits b = base_b_107 b → b = 6 :=
by sorry

end NUMINAMATH_CALUDE_base_is_six_l2888_288895


namespace NUMINAMATH_CALUDE_smallest_k_for_positive_c_l2888_288880

theorem smallest_k_for_positive_c (a b c k : ℤ) : 
  a < b → b < c → 
  (2 * b = a + c) →  -- arithmetic progression
  (k * c)^2 = a * b →  -- geometric progression
  k > 1 → 
  c > 0 → 
  (∀ m : ℤ, m > 1 → m < k → ¬(∃ a' b' c' : ℤ, 
    a' < b' ∧ b' < c' ∧ 
    (2 * b' = a' + c') ∧ 
    (m * c')^2 = a' * b' ∧ 
    c' > 0)) → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_for_positive_c_l2888_288880


namespace NUMINAMATH_CALUDE_max_congruent_triangles_l2888_288881

-- Define a point in the plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a triangle
structure Triangle :=
  (p1 : Point) (p2 : Point) (p3 : Point)

-- Define congruence of triangles
def CongruentTriangles (t1 t2 : Triangle) : Prop :=
  sorry

-- Define the main theorem
theorem max_congruent_triangles
  (A B C D : Point)
  (h : A.x * B.y - A.y * B.x ≠ C.x * D.y - C.y * D.x) :
  (∃ (n : ℕ), ∀ (m : ℕ), 
    (∃ (X : Fin m → Point), 
      (∀ (i : Fin m), CongruentTriangles 
        (Triangle.mk A B (X i)) 
        (Triangle.mk C D (X i)))) → 
    m ≤ n) ∧
  (∃ (X : Fin 4 → Point), 
    (∀ (i : Fin 4), CongruentTriangles 
      (Triangle.mk A B (X i)) 
      (Triangle.mk C D (X i)))) :=
sorry


end NUMINAMATH_CALUDE_max_congruent_triangles_l2888_288881


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2888_288871

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_z : ℝ), min_z = 5 ∧ ∀ z : ℝ, z = 5 * x^2 + 20 * x + 25 → z ≥ min_z :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2888_288871


namespace NUMINAMATH_CALUDE_ngon_triangle_division_l2888_288842

/-- 
Given an n-gon divided into k triangles, prove that k ≥ n-2.
-/
theorem ngon_triangle_division (n k : ℕ) (h1 : n ≥ 3) (h2 : k > 0) : k ≥ n - 2 := by
  sorry


end NUMINAMATH_CALUDE_ngon_triangle_division_l2888_288842


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l2888_288859

/-- The constant term in the expansion of (3x + 2/x)^8 is 90720 -/
theorem constant_term_of_expansion (x : ℝ) (x_ne_zero : x ≠ 0) : 
  (Finset.range 9).sum (λ k => Nat.choose 8 k * (3^(8-k) * 2^k * x^(8-2*k))) = 90720 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l2888_288859


namespace NUMINAMATH_CALUDE_intersection_x_axis_intersection_y_axis_l2888_288889

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 2 * x - 1

-- Theorem for the intersection with x-axis
theorem intersection_x_axis :
  ∃ (x : ℝ), line_equation x 0 ∧ x = 0.5 := by sorry

-- Theorem for the intersection with y-axis
theorem intersection_y_axis :
  line_equation 0 (-1) := by sorry

end NUMINAMATH_CALUDE_intersection_x_axis_intersection_y_axis_l2888_288889


namespace NUMINAMATH_CALUDE_dice_probability_l2888_288815

def num_dice : ℕ := 6
def sides_per_die : ℕ := 18
def one_digit_numbers : ℕ := 9
def two_digit_numbers : ℕ := 9

theorem dice_probability :
  let p_one_digit : ℚ := one_digit_numbers / sides_per_die
  let p_two_digit : ℚ := two_digit_numbers / sides_per_die
  let choose_three : ℕ := Nat.choose num_dice 3
  choose_three * (p_one_digit ^ 3 * p_two_digit ^ 3) = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_dice_probability_l2888_288815


namespace NUMINAMATH_CALUDE_new_group_weight_calculation_l2888_288882

/-- The total combined weight of 5 new people replacing 5 others in a group of 20 --/
def new_group_weight (initial_group_size : ℕ) (replaced_weights : List ℝ) (average_increase : ℝ) : ℝ :=
  (initial_group_size * average_increase) + replaced_weights.sum

theorem new_group_weight_calculation :
  let initial_group_size : ℕ := 20
  let replaced_weights : List ℝ := [40, 55, 60, 75, 80]
  let average_increase : ℝ := 2
  new_group_weight initial_group_size replaced_weights average_increase = 350 := by
  sorry

end NUMINAMATH_CALUDE_new_group_weight_calculation_l2888_288882


namespace NUMINAMATH_CALUDE_impossibleStar_l2888_288866

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the star vertices
variable (A1 A2 A3 A4 A5 : Point3D)

-- Define a function to check if a point is above a plane
def isAbovePlane (p : Point3D) (p1 p2 p3 : Point3D) : Prop := sorry

-- Define a function to check if a point is below a plane
def isBelowPlane (p : Point3D) (p1 p2 p3 : Point3D) : Prop := sorry

-- Define a function to check if two points are connected by a straight line
def areConnected (p1 p2 : Point3D) : Prop := sorry

-- Theorem statement
theorem impossibleStar (h1 : isAbovePlane A2 A1 A3 A5)
                       (h2 : isBelowPlane A4 A1 A3 A5)
                       (h3 : areConnected A2 A4) :
  False := sorry

end NUMINAMATH_CALUDE_impossibleStar_l2888_288866


namespace NUMINAMATH_CALUDE_jills_favorite_number_hint_l2888_288821

def is_even (n : Nat) : Prop := ∃ k, n = 2 * k

def has_repeating_prime_factors (n : Nat) : Prop :=
  ∃ p : Nat, Nat.Prime p ∧ (∃ k > 1, p ^ k ∣ n)

def best_guess : Nat := 84

theorem jills_favorite_number_hint :
  ∀ n : Nat,
  is_even n →
  has_repeating_prime_factors n →
  best_guess = 84 →
  (∃ p : Nat, Nat.Prime p ∧ (∃ k > 1, p ^ k ∣ n) ∧ p = 2) :=
by sorry

end NUMINAMATH_CALUDE_jills_favorite_number_hint_l2888_288821


namespace NUMINAMATH_CALUDE_max_similar_triangle_pairs_l2888_288847

-- Define the triangle ABC
variable (A B C : Point)

-- Define that ABC is acute-angled
def is_acute_angled (A B C : Point) : Prop := sorry

-- Define heights AL and BM
def height_AL (A L : Point) : Prop := sorry
def height_BM (B M : Point) : Prop := sorry

-- Define that LM intersects the extension of AB at point D
def LM_intersects_AB_extension (L M D : Point) : Prop := sorry

-- Define a function to count similar triangle pairs
def count_similar_triangle_pairs (A B C L M D : Point) : ℕ := sorry

-- Define that no pairs of congruent triangles are formed
def no_congruent_triangles (A B C L M D : Point) : Prop := sorry

theorem max_similar_triangle_pairs 
  (A B C L M D : Point) 
  (h1 : is_acute_angled A B C)
  (h2 : height_AL A L)
  (h3 : height_BM B M)
  (h4 : LM_intersects_AB_extension L M D)
  (h5 : no_congruent_triangles A B C L M D) :
  count_similar_triangle_pairs A B C L M D = 10 := by sorry

end NUMINAMATH_CALUDE_max_similar_triangle_pairs_l2888_288847


namespace NUMINAMATH_CALUDE_total_tiles_count_l2888_288829

def room_length : ℕ := 24
def room_width : ℕ := 18
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

def border_tiles : ℕ :=
  2 * (room_length / border_tile_size + room_width / border_tile_size) - 4

def inner_area : ℕ :=
  (room_length - 2 * border_tile_size) * (room_width - 2 * border_tile_size)

def inner_tiles : ℕ :=
  (inner_area + inner_tile_size^2 - 1) / inner_tile_size^2

theorem total_tiles_count :
  border_tiles + inner_tiles = 70 := by sorry

end NUMINAMATH_CALUDE_total_tiles_count_l2888_288829


namespace NUMINAMATH_CALUDE_evaluate_expression_l2888_288887

theorem evaluate_expression (x y : ℝ) (hx : x = 4) (hy : y = 9) :
  2 * x^(y/2) + 5 * y^(x/2) = 1429 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2888_288887


namespace NUMINAMATH_CALUDE_father_son_ages_new_age_ratio_l2888_288861

/-- Given the ratio of a father's age to his son's age and their age product, 
    prove the father's age, son's age, and their combined income. -/
theorem father_son_ages (father_son_ratio : ℚ) (age_product : ℕ) (income_percentage : ℚ) :
  father_son_ratio = 7/3 →
  age_product = 756 →
  income_percentage = 2/5 →
  ∃ (father_age son_age : ℕ) (combined_income : ℚ),
    father_age = 42 ∧
    son_age = 18 ∧
    combined_income = 105 ∧
    (father_age : ℚ) / son_age = father_son_ratio ∧
    father_age * son_age = age_product ∧
    (father_age : ℚ) = income_percentage * combined_income :=
by sorry

/-- Given the father's and son's ages, prove their new age ratio after 6 years. -/
theorem new_age_ratio (father_age son_age : ℕ) (years : ℕ) :
  father_age = 42 →
  son_age = 18 →
  years = 6 →
  ∃ (new_ratio : ℚ),
    new_ratio = 2/1 ∧
    new_ratio = (father_age + years : ℚ) / (son_age + years) :=
by sorry

end NUMINAMATH_CALUDE_father_son_ages_new_age_ratio_l2888_288861


namespace NUMINAMATH_CALUDE_election_win_percentage_l2888_288858

theorem election_win_percentage 
  (total_votes : ℕ)
  (geoff_percentage : ℚ)
  (additional_votes_needed : ℕ)
  (h1 : total_votes = 6000)
  (h2 : geoff_percentage = 1/200)  -- 0.5% as a rational number
  (h3 : additional_votes_needed = 3000) :
  (((geoff_percentage * total_votes).floor + additional_votes_needed : ℚ) / total_votes) * 100 = 505/10 :=
sorry

end NUMINAMATH_CALUDE_election_win_percentage_l2888_288858


namespace NUMINAMATH_CALUDE_two_arithmetic_sequences_sum_l2888_288833

def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem two_arithmetic_sequences_sum : 
  let seq1_sum := arithmetic_sum 2 10 5
  let seq2_sum := arithmetic_sum 10 10 5
  seq1_sum + seq2_sum = 260 := by
  sorry

end NUMINAMATH_CALUDE_two_arithmetic_sequences_sum_l2888_288833


namespace NUMINAMATH_CALUDE_plumber_copper_pipe_l2888_288822

/-- The number of meters of copper pipe bought by the plumber -/
def copper_pipe : ℕ := sorry

/-- The number of meters of plastic pipe bought by the plumber -/
def plastic_pipe : ℕ := sorry

/-- The cost of one meter of pipe in dollars -/
def cost_per_meter : ℕ := 4

/-- The total cost of all pipes in dollars -/
def total_cost : ℕ := 100

theorem plumber_copper_pipe :
  copper_pipe = 10 ∧
  plastic_pipe = copper_pipe + 5 ∧
  cost_per_meter * (copper_pipe + plastic_pipe) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_plumber_copper_pipe_l2888_288822


namespace NUMINAMATH_CALUDE_stating_max_valid_pairs_l2888_288827

/-- Represents the maximum value that can be used in the pairs -/
def maxValue : ℕ := 2018

/-- Represents a pair of natural numbers (a, b) where a < b ≤ maxValue -/
structure ValidPair where
  a : ℕ
  b : ℕ
  h1 : a < b
  h2 : b ≤ maxValue

/-- Represents a set of valid pairs satisfying the given conditions -/
def ValidPairSet := Set ValidPair

/-- 
  Given a set of valid pairs, returns the number of pairs in the set
  satisfying the condition that if (a, b) is in the set, 
  then (c, a) and (b, d) are not in the set for any c and d
-/
def countValidPairs (s : ValidPairSet) : ℕ := sorry

/-- The maximum number of valid pairs that can be written on the board -/
def maxPairs : ℕ := 1018081

/-- 
  Theorem stating that the maximum number of valid pairs 
  that can be written on the board is 1018081
-/
theorem max_valid_pairs : 
  ∀ s : ValidPairSet, countValidPairs s ≤ maxPairs ∧ 
  ∃ s : ValidPairSet, countValidPairs s = maxPairs := by sorry

end NUMINAMATH_CALUDE_stating_max_valid_pairs_l2888_288827


namespace NUMINAMATH_CALUDE_max_value_of_expr_l2888_288876

def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def expr (a b c : ℕ) : ℚ := 1 / (a + 2010 / (b + 1 / c))

theorem max_value_of_expr (a b c : ℕ) 
  (ha : is_nonzero_digit a) 
  (hb : is_nonzero_digit b) 
  (hc : is_nonzero_digit c) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  expr a b c ≤ 1 / 203 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expr_l2888_288876


namespace NUMINAMATH_CALUDE_chord_equation_of_parabola_l2888_288857

/-- Given a parabola y² = 4x and a chord with midpoint (1, 1),
    the equation of the line containing this chord is 2x - y - 1 = 0 -/
theorem chord_equation_of_parabola (x y : ℝ) :
  (y^2 = 4*x) →  -- parabola equation
  ∃ (x1 y1 x2 y2 : ℝ),
    (y1^2 = 4*x1) ∧ (y2^2 = 4*x2) ∧  -- points on parabola
    ((x1 + x2)/2 = 1) ∧ ((y1 + y2)/2 = 1) ∧  -- midpoint condition
    (2*x - y - 1 = 0) →  -- equation of the line
  ∃ (k : ℝ), y - 1 = k*(x - 1) ∧ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_of_parabola_l2888_288857


namespace NUMINAMATH_CALUDE_f_2011_eq_sin_l2888_288860

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => Real.cos
| (n + 1) => deriv (f n)

theorem f_2011_eq_sin : f 2011 = Real.sin := by sorry

end NUMINAMATH_CALUDE_f_2011_eq_sin_l2888_288860


namespace NUMINAMATH_CALUDE_robert_read_315_pages_l2888_288803

/-- The number of pages Robert read in the book over 10 days -/
def total_pages (days_1 days_2 days_3 : ℕ) 
                (pages_per_day_1 pages_per_day_2 pages_day_3 : ℕ) : ℕ :=
  days_1 * pages_per_day_1 + days_2 * pages_per_day_2 + pages_day_3

/-- Theorem stating that Robert read 315 pages in total -/
theorem robert_read_315_pages : 
  total_pages 5 4 1 25 40 30 = 315 := by
  sorry

#eval total_pages 5 4 1 25 40 30

end NUMINAMATH_CALUDE_robert_read_315_pages_l2888_288803


namespace NUMINAMATH_CALUDE_number_plus_eight_equals_500_l2888_288838

theorem number_plus_eight_equals_500 (x : ℤ) : x + 8 = 500 → x = 492 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_eight_equals_500_l2888_288838


namespace NUMINAMATH_CALUDE_circle_configuration_exists_l2888_288874

/-- A configuration of numbers in circles -/
structure CircleConfiguration where
  numbers : Fin 9 → ℕ
  consecutive : ∀ i j : Fin 9, i.val < j.val → numbers i < numbers j
  contains_six : ∃ i : Fin 9, numbers i = 6

/-- The lines connecting the circles -/
inductive Line
  | Line1 : Line
  | Line2 : Line
  | Line3 : Line
  | Line4 : Line
  | Line5 : Line
  | Line6 : Line

/-- The endpoints of each line -/
def lineEndpoints : Line → Fin 9 × Fin 9
  | Line.Line1 => (⟨0, by norm_num⟩, ⟨1, by norm_num⟩)
  | Line.Line2 => (⟨1, by norm_num⟩, ⟨2, by norm_num⟩)
  | Line.Line3 => (⟨2, by norm_num⟩, ⟨3, by norm_num⟩)
  | Line.Line4 => (⟨3, by norm_num⟩, ⟨4, by norm_num⟩)
  | Line.Line5 => (⟨4, by norm_num⟩, ⟨5, by norm_num⟩)
  | Line.Line6 => (⟨5, by norm_num⟩, ⟨0, by norm_num⟩)

/-- The sum of numbers on a line -/
def lineSum (config : CircleConfiguration) (line : Line) : ℕ :=
  let (a, b) := lineEndpoints line
  config.numbers a + config.numbers b

/-- The theorem statement -/
theorem circle_configuration_exists :
  ∃ config : CircleConfiguration, ∀ line : Line, lineSum config line = 23 := by
  sorry


end NUMINAMATH_CALUDE_circle_configuration_exists_l2888_288874


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2888_288834

/-- Given a quadratic function f(x) = ax^2 + bx + c, 
    if f(x₁) = f(x₂) and x₁ ≠ x₂, then f(x₁ + x₂) = c -/
theorem quadratic_function_property (a b c x₁ x₂ : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  x₁ ≠ x₂ → f x₁ = f x₂ → f (x₁ + x₂) = c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2888_288834


namespace NUMINAMATH_CALUDE_mutated_frogs_percentage_l2888_288850

def total_frogs : ℕ := 27
def mutated_frogs : ℕ := 9

def percentage_mutated : ℚ := (mutated_frogs : ℚ) / (total_frogs : ℚ) * 100

def rounded_percentage : ℕ := 
  (percentage_mutated + 0.5).floor.toNat

theorem mutated_frogs_percentage :
  rounded_percentage = 33 := by sorry

end NUMINAMATH_CALUDE_mutated_frogs_percentage_l2888_288850


namespace NUMINAMATH_CALUDE_expression_equals_seventy_percent_l2888_288824

theorem expression_equals_seventy_percent (y : ℝ) (c : ℝ) (h1 : y > 0) 
  (h2 : (8 * y) / 20 + (c * y) / 10 = 0.7 * y) : c = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seventy_percent_l2888_288824


namespace NUMINAMATH_CALUDE_amount_calculation_l2888_288813

/-- Given three people with a total amount of money, where one person has a specific fraction of the others' total, calculate that person's amount. -/
theorem amount_calculation (total : ℚ) (p q r : ℚ) : 
  total = 5000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 2000 := by
  sorry

end NUMINAMATH_CALUDE_amount_calculation_l2888_288813


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2888_288845

def set_A : Set ℝ := {x | Real.sqrt x ≤ 3}
def set_B : Set ℝ := {x | x^2 ≤ 9}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | 0 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2888_288845


namespace NUMINAMATH_CALUDE_sum_of_periodic_function_l2888_288854

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

def translate_right (f : ℝ → ℝ) (n : ℝ) : ℝ → ℝ :=
  fun x ↦ f (x - n)

theorem sum_of_periodic_function
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_odd_translated : is_odd_function (translate_right f 1))
  (h_f2 : f 2 = -1) :
  (Finset.range 2011).sum (fun i ↦ f (i + 1 : ℝ)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_periodic_function_l2888_288854


namespace NUMINAMATH_CALUDE_sqrt_of_point_zero_one_l2888_288846

theorem sqrt_of_point_zero_one : Real.sqrt 0.01 = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_point_zero_one_l2888_288846


namespace NUMINAMATH_CALUDE_room_population_l2888_288852

theorem room_population (P : ℕ) 
  (women_ratio : (2 : ℚ) / 5 * P = (P : ℚ).floor)
  (married_ratio : (1 : ℚ) / 2 * P = (P : ℚ).floor)
  (max_unmarried_women : ℕ → Prop)
  (h_max_unmarried : max_unmarried_women 32) :
  P = 64 := by
sorry

end NUMINAMATH_CALUDE_room_population_l2888_288852


namespace NUMINAMATH_CALUDE_square_area_ratio_l2888_288835

theorem square_area_ratio (side_c side_d : ℝ) (h1 : side_c = 48) (h2 : side_d = 60) :
  (side_c^2) / (side_d^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2888_288835


namespace NUMINAMATH_CALUDE_ratio_of_a_to_c_l2888_288899

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_c_l2888_288899


namespace NUMINAMATH_CALUDE_trapezoid_angle_sum_l2888_288877

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides -/
structure Trapezoid where
  angles : Fin 4 → ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360
  parallel_sides : ∃ (i j : Fin 4), i ≠ j ∧ (angles i) + (angles j) = 180

/-- Given a trapezoid with two angles of 60° and 120°, the sum of the other two angles is 180° -/
theorem trapezoid_angle_sum (t : Trapezoid) 
  (h1 : ∃ (i : Fin 4), t.angles i = 60)
  (h2 : ∃ (j : Fin 4), t.angles j = 120) :
  ∃ (k l : Fin 4), k ≠ l ∧ t.angles k + t.angles l = 180 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_angle_sum_l2888_288877


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l2888_288832

theorem half_angle_quadrant (α : Real) (k : Int) : 
  (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3/2 * Real.pi) →
  ((∃ n : Int, 2 * n * Real.pi + Real.pi/2 < α/2 ∧ α/2 < 2 * n * Real.pi + 3/4 * Real.pi) ∨
   (∃ n : Int, (2 * n + 1) * Real.pi + Real.pi/2 < α/2 ∧ α/2 < (2 * n + 1) * Real.pi + 3/4 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l2888_288832


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2888_288867

theorem inequality_system_solution_set (a : ℝ) : 
  (∃ x : ℝ, a * x > -1 ∧ x + a > 0) ↔ a > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2888_288867


namespace NUMINAMATH_CALUDE_decimal_189_to_base_4_lsd_l2888_288856

def decimal_to_base_4_lsd (n : ℕ) : ℕ :=
  n % 4

theorem decimal_189_to_base_4_lsd :
  decimal_to_base_4_lsd 189 = 1 := by
  sorry

end NUMINAMATH_CALUDE_decimal_189_to_base_4_lsd_l2888_288856


namespace NUMINAMATH_CALUDE_original_expenditure_is_420_l2888_288808

/-- Calculates the original expenditure of a student mess given the following conditions:
  * There were initially 35 students
  * 7 new students were admitted
  * The total expenses increased by 42 rupees per day
  * The average expenditure per head decreased by 1 rupee
-/
def calculate_original_expenditure (initial_students : ℕ) (new_students : ℕ) 
  (expense_increase : ℕ) (average_decrease : ℕ) : ℕ :=
  let total_students := initial_students + new_students
  let x := (expense_increase + total_students * average_decrease) / (total_students - initial_students)
  initial_students * x

/-- Theorem stating that under the given conditions, the original expenditure was 420 rupees per day -/
theorem original_expenditure_is_420 :
  calculate_original_expenditure 35 7 42 1 = 420 := by
  sorry

end NUMINAMATH_CALUDE_original_expenditure_is_420_l2888_288808


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_seven_undefined_inverse_smallest_a_is_seven_l2888_288864

theorem smallest_undefined_inverse (a : ℕ) : a > 0 ∧ 
  ¬ (∃ x : ℕ, x * a ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ y : ℕ, y * a ≡ 1 [MOD 77]) →
  a ≥ 7 :=
by sorry

theorem seven_undefined_inverse : 
  ¬ (∃ x : ℕ, x * 7 ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ y : ℕ, y * 7 ≡ 1 [MOD 77]) :=
by sorry

theorem smallest_a_is_seven : 
  ∃ a : ℕ, a > 0 ∧
  ¬ (∃ x : ℕ, x * a ≡ 1 [MOD 70]) ∧
  ¬ (∃ y : ℕ, y * a ≡ 1 [MOD 77]) ∧
  ∀ b : ℕ, b > 0 ∧ 
    ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 70]) ∧ 
    ¬ (∃ y : ℕ, y * b ≡ 1 [MOD 77]) →
    b ≥ a :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_seven_undefined_inverse_smallest_a_is_seven_l2888_288864


namespace NUMINAMATH_CALUDE_find_number_to_multiply_l2888_288879

theorem find_number_to_multiply : ∃ x : ℤ, 43 * x - 34 * x = 1251 :=
by sorry

end NUMINAMATH_CALUDE_find_number_to_multiply_l2888_288879


namespace NUMINAMATH_CALUDE_xiao_ming_test_average_l2888_288819

theorem xiao_ming_test_average (first_two_avg : ℝ) (last_three_total : ℝ) :
  first_two_avg = 85 →
  last_three_total = 270 →
  (2 * first_two_avg + last_three_total) / 5 = 88 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_test_average_l2888_288819


namespace NUMINAMATH_CALUDE_five_distinct_values_l2888_288818

/-- The number of distinct values obtainable by rearranging parentheses in 3^(3^(3^3)) -/
def num_distinct_values : ℕ := 5

/-- The original expression 3^(3^(3^3)) -/
def original_expression : ℕ := 3^(3^(3^3))

/-- All possible parenthesizations of 3^3^3^3 -/
def all_parenthesizations : List (ℕ → ℕ → ℕ → ℕ) :=
  [ (fun a b c => a^(b^c)),
    (fun a b c => a^((b^c))),
    (fun a b c => ((a^b)^c)),
    (fun a b c => (a^(b^c))),
    (fun a b c => (a^b)^c) ]

/-- Theorem stating that there are exactly 5 distinct values obtainable -/
theorem five_distinct_values :
  (List.map (fun f => f 3 3 3) all_parenthesizations).toFinset.card = num_distinct_values :=
sorry

end NUMINAMATH_CALUDE_five_distinct_values_l2888_288818


namespace NUMINAMATH_CALUDE_soccer_points_for_win_l2888_288891

theorem soccer_points_for_win (total_games : ℕ) (wins : ℕ) (losses : ℕ) (total_points : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_total_points : total_points = 46)
  (h_games_balance : total_games = wins + losses + (total_games - wins - losses)) :
  ∃ (points_for_win : ℕ),
    points_for_win * wins + (total_games - wins - losses) = total_points ∧ 
    points_for_win = 3 := by
sorry

end NUMINAMATH_CALUDE_soccer_points_for_win_l2888_288891


namespace NUMINAMATH_CALUDE_philips_banana_collection_l2888_288868

/-- The number of groups of bananas in Philip's collection -/
def num_groups : ℕ := 196

/-- The number of bananas in each group -/
def bananas_per_group : ℕ := 2

/-- The total number of bananas in Philip's collection -/
def total_bananas : ℕ := num_groups * bananas_per_group

theorem philips_banana_collection : total_bananas = 392 := by
  sorry

end NUMINAMATH_CALUDE_philips_banana_collection_l2888_288868


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l2888_288851

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ := (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem four_digit_integer_problem (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : digit_sum n = 16)
  (h3 : middle_digits_sum n = 10)
  (h4 : thousands_minus_units n = 2)
  (h5 : n % 9 = 0) :
  n = 4522 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l2888_288851


namespace NUMINAMATH_CALUDE_right_triangle_side_relation_l2888_288885

theorem right_triangle_side_relation (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b = c * x →
  1 < x ∧ x ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_relation_l2888_288885


namespace NUMINAMATH_CALUDE_zoo_animals_l2888_288898

theorem zoo_animals (birds : ℕ) (non_birds : ℕ) : 
  birds = 450 → 
  birds = 5 * non_birds → 
  birds - non_birds = 360 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l2888_288898


namespace NUMINAMATH_CALUDE_power_of_power_l2888_288810

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2888_288810


namespace NUMINAMATH_CALUDE_min_value_theorem_l2888_288875

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_seq : x * Real.log 2 + y * Real.log 2 = 2 * Real.log (Real.sqrt 2)) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a * Real.log 2 + b * Real.log 2 = 2 * Real.log (Real.sqrt 2) →
  1 / x + 9 / y ≤ 1 / a + 9 / b :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2888_288875


namespace NUMINAMATH_CALUDE_tan_315_degrees_l2888_288825

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l2888_288825


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l2888_288894

theorem sum_of_squares_and_products (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x^2 + y^2 + z^2 = 52 → x*y + y*z + z*x = 24 → x + y + z = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l2888_288894


namespace NUMINAMATH_CALUDE_cannot_buy_without_change_l2888_288855

theorem cannot_buy_without_change (zloty_to_grosz : ℕ) (total_zloty : ℕ) (item_price_grosz : ℕ) :
  zloty_to_grosz = 1001 →
  total_zloty = 1986 →
  item_price_grosz = 1987 →
  ¬ (∃ n : ℕ, n * item_price_grosz = total_zloty * zloty_to_grosz) :=
by sorry

end NUMINAMATH_CALUDE_cannot_buy_without_change_l2888_288855


namespace NUMINAMATH_CALUDE_max_candies_eaten_l2888_288812

/-- Represents the state of the board and the total candies eaten -/
structure BoardState :=
  (numbers : List ℕ)
  (candies : ℕ)

/-- Represents one step of Karlson's process -/
def step (state : BoardState) : BoardState :=
  sorry

/-- The initial state of the board -/
def initial_state : BoardState :=
  { numbers := List.replicate 40 1, candies := 0 }

/-- Applies the step function n times -/
def apply_n_steps (n : ℕ) (state : BoardState) : BoardState :=
  sorry

theorem max_candies_eaten :
  ∃ (final_state : BoardState),
    final_state = apply_n_steps 40 initial_state ∧
    final_state.candies ≤ 780 ∧
    ∀ (other_final_state : BoardState),
      other_final_state = apply_n_steps 40 initial_state →
      other_final_state.candies ≤ final_state.candies :=
sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l2888_288812


namespace NUMINAMATH_CALUDE_greatest_n_value_greatest_n_is_10_l2888_288853

theorem greatest_n_value (n : ℤ) (h : 303 * n^3 ≤ 380000) : n ≤ 10 := by
  sorry

theorem greatest_n_is_10 : ∃ n : ℤ, 303 * n^3 ≤ 380000 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_greatest_n_is_10_l2888_288853


namespace NUMINAMATH_CALUDE_sticker_exchange_result_l2888_288844

/-- The total number of stickers after the exchange event -/
def total_stickers_after_exchange (ryan steven terry emily jasmine : ℕ) : ℕ :=
  ryan + steven + terry + emily + jasmine - 5 * 2

theorem sticker_exchange_result :
  let ryan := 30
  let steven := 3 * ryan
  let terry := steven + 20
  let emily := steven / 2
  let jasmine := terry + terry / 10
  total_stickers_after_exchange ryan steven terry emily jasmine = 386 := by
  sorry

#eval total_stickers_after_exchange 30 90 110 45 121

end NUMINAMATH_CALUDE_sticker_exchange_result_l2888_288844


namespace NUMINAMATH_CALUDE_tetrahedron_face_areas_sum_squares_tetrahedron_face_areas_volume_inequality_l2888_288837

-- Define the tetrahedron structure
structure Tetrahedron where
  V : ℝ  -- Volume
  S_A : ℝ  -- Face area opposite to vertex A
  S_B : ℝ  -- Face area opposite to vertex B
  S_C : ℝ  -- Face area opposite to vertex C
  S_D : ℝ  -- Face area opposite to vertex D
  a : ℝ   -- Length of edge BC
  a' : ℝ  -- Length of edge DA
  b : ℝ   -- Length of edge CA
  b' : ℝ  -- Length of edge DB
  c : ℝ   -- Length of edge AB
  c' : ℝ  -- Length of edge DC
  α : ℝ   -- Angle between opposite edges BC and DA
  β : ℝ   -- Angle between opposite edges CA and DB
  γ : ℝ   -- Angle between opposite edges AB and DC

-- Theorem statements
theorem tetrahedron_face_areas_sum_squares (t : Tetrahedron) :
  t.S_A^2 + t.S_B^2 + t.S_C^2 + t.S_D^2 = 
    1/4 * ((t.a * t.a' * Real.sin t.α)^2 + 
           (t.b * t.b' * Real.sin t.β)^2 + 
           (t.c * t.c' * Real.sin t.γ)^2) :=
  sorry

theorem tetrahedron_face_areas_volume_inequality (t : Tetrahedron) :
  t.S_A^2 + t.S_B^2 + t.S_C^2 + t.S_D^2 ≥ 9 * (3 * t.V^4)^(1/3) :=
  sorry

end NUMINAMATH_CALUDE_tetrahedron_face_areas_sum_squares_tetrahedron_face_areas_volume_inequality_l2888_288837


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2888_288848

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x ∧ x = Real.sqrt (1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2888_288848


namespace NUMINAMATH_CALUDE_find_c_l2888_288809

/-- Given two functions p and q, prove that c = 7 -/
theorem find_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 3 * x - 9) → 
  (∀ x, q x = 4 * x - c) → 
  p (q 3) = 6 → 
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_find_c_l2888_288809


namespace NUMINAMATH_CALUDE_train_length_calculation_l2888_288823

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
theorem train_length_calculation (jogger_speed train_speed : ℝ) (initial_distance passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →
  train_speed = 45 * (5 / 18) →
  initial_distance = 180 →
  passing_time = 30 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 120 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l2888_288823


namespace NUMINAMATH_CALUDE_gcd_8157_2567_l2888_288816

theorem gcd_8157_2567 : Nat.gcd 8157 2567 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8157_2567_l2888_288816


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2888_288831

theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k^2 - 3 > 0) → (k > 2 ∨ k < -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2888_288831


namespace NUMINAMATH_CALUDE_remaining_volume_of_cone_volume_of_remaining_part_is_27_l2888_288820

/-- Represents a cone with an inscribed sphere and a plane through the circle of tangency -/
structure InscribedSphereCone where
  /-- The angle between the slant height and the base plane -/
  α : Real
  /-- The volume of the part of the cone enclosed between the tangency plane and the base plane -/
  enclosed_volume : Real

/-- Theorem stating the volume of the remaining part of the cone -/
theorem remaining_volume_of_cone (cone : InscribedSphereCone) 
  (h1 : cone.α = Real.arccos (1/4))
  (h2 : cone.enclosed_volume = 37) :
  64 - cone.enclosed_volume = 27 := by
  sorry

/-- Main theorem to prove -/
theorem volume_of_remaining_part_is_27 (cone : InscribedSphereCone) 
  (h1 : cone.α = Real.arccos (1/4))
  (h2 : cone.enclosed_volume = 37) :
  ∃ (v : Real), v = 27 ∧ v = 64 - cone.enclosed_volume := by
  sorry

end NUMINAMATH_CALUDE_remaining_volume_of_cone_volume_of_remaining_part_is_27_l2888_288820


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l2888_288883

theorem monic_quadratic_with_complex_root :
  let p : ℂ → ℂ := fun x ↦ x^2 - 6*x + 25
  (∀ x : ℂ, p x = 0 → x = 3 - 4*I ∨ x = 3 + 4*I) ∧
  (p (3 - 4*I) = 0) ∧
  (∀ x : ℝ, p x = x^2 - 6*x + 25) :=
by sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l2888_288883


namespace NUMINAMATH_CALUDE_triangle_distance_inequality_l2888_288873

/-- Given a triangle ABC with an internal point P, prove the inequality involving distances from P to vertices and sides. -/
theorem triangle_distance_inequality 
  (x y z p q r : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hp : p ≥ 0) (hq : q ≥ 0) (hr : r ≥ 0) 
  (h_internal : x + y > z ∧ y + z > x ∧ z + x > y) : 
  x * y * z ≥ (q + r) * (r + p) * (p + q) := by
  sorry

end NUMINAMATH_CALUDE_triangle_distance_inequality_l2888_288873


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_S_l2888_288826

def U : Set ℕ := Set.univ

def S : Set ℕ := {x ∈ U | x^2 - x = 0}

def T : Set ℕ := {x ∈ U | ∃ k : ℤ, 6 = k * (x - 2)}

theorem S_intersect_T_eq_S : S ∩ T = S := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_S_l2888_288826


namespace NUMINAMATH_CALUDE_range_of_m_l2888_288817

theorem range_of_m (x m : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 10 → 1 - m ≤ x ∧ x ≤ 1 + m) ∧ 
  (∃ x, 1 - m ≤ x ∧ x ≤ 1 + m ∧ (x < -2 ∨ x > 10)) → 
  m ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2888_288817


namespace NUMINAMATH_CALUDE_compute_expression_l2888_288800

theorem compute_expression : (-3) * 2 + 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2888_288800


namespace NUMINAMATH_CALUDE_certain_number_minus_fifteen_l2888_288890

theorem certain_number_minus_fifteen (x : ℝ) : x / 10 = 6 → x - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_minus_fifteen_l2888_288890


namespace NUMINAMATH_CALUDE_faulty_faucet_leak_l2888_288869

/-- The amount of water leaked by a faulty faucet in half an hour -/
def water_leaked (leak_rate : ℝ) (time : ℝ) : ℝ :=
  leak_rate * time

theorem faulty_faucet_leak : 
  let leak_rate : ℝ := 65  -- grams per minute
  let time : ℝ := 30       -- half an hour in minutes
  water_leaked leak_rate time = 1950 := by
sorry

end NUMINAMATH_CALUDE_faulty_faucet_leak_l2888_288869


namespace NUMINAMATH_CALUDE_complex_quadrant_l2888_288830

theorem complex_quadrant (z : ℂ) (h : (3 + 4*I)*z = 25) : 
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_quadrant_l2888_288830


namespace NUMINAMATH_CALUDE_perpendicular_lines_imply_a_equals_one_l2888_288802

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_imply_a_equals_one :
  ∀ (a l : ℝ),
  let line1 : Line := { a := a + l, b := 2, c := 0 }
  let line2 : Line := { a := 1, b := -a, c := -1 }
  perpendicular line1 line2 → a = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_imply_a_equals_one_l2888_288802


namespace NUMINAMATH_CALUDE_cosine_symmetry_center_l2888_288839

/-- The symmetry center of the cosine function f(x) = 3cos(2(x - π/6) + π/2) -/
theorem cosine_symmetry_center : 
  let f : ℝ → ℝ := λ x ↦ 3 * Real.cos (2 * (x - π/6) + π/2)
  ∃ (center : ℝ × ℝ), center = (π/6, 0) ∧ 
    ∀ (x : ℝ), f (center.1 + x) = f (center.1 - x) :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_center_l2888_288839


namespace NUMINAMATH_CALUDE_least_positive_integer_y_l2888_288806

theorem least_positive_integer_y (y : ℕ) : 
  (∀ k : ℕ, 0 < k ∧ k < 4 → ¬(53 ∣ (3*k)^2 + 3*41*3*k + 41^2)) ∧ 
  (53 ∣ (3*4)^2 + 3*41*3*4 + 41^2) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_y_l2888_288806


namespace NUMINAMATH_CALUDE_decimal_equivalences_l2888_288801

theorem decimal_equivalences (d : ℚ) (h : d = 0.25) : 
  d = 3 / 12 ∧ d = 8 / 32 ∧ d = 25 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalences_l2888_288801


namespace NUMINAMATH_CALUDE_x_range_l2888_288804

theorem x_range (x : ℝ) : 
  (Real.log (x^2 - 2*x - 2) ≥ 0) → 
  ¬(0 < x ∧ x < 4) → 
  (x ≥ 4 ∨ x ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l2888_288804


namespace NUMINAMATH_CALUDE_max_value_of_a_l2888_288878

theorem max_value_of_a (a b c : ℕ+) 
  (h : a + b + c = Nat.gcd a b + Nat.gcd b c + Nat.gcd c a + 120) :
  a ≤ 240 ∧ ∃ a₀ b₀ c₀ : ℕ+, a₀ = 240 ∧ 
    a₀ + b₀ + c₀ = Nat.gcd a₀ b₀ + Nat.gcd b₀ c₀ + Nat.gcd c₀ a₀ + 120 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2888_288878


namespace NUMINAMATH_CALUDE_sock_pairs_count_l2888_288805

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white brown blue black : ℕ) : ℕ :=
  white * brown + white * blue + white * black +
  brown * blue + brown * black +
  blue * black

/-- Theorem stating the number of ways to choose a pair of socks of different colors -/
theorem sock_pairs_count :
  differentColorPairs 5 5 3 3 = 94 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l2888_288805


namespace NUMINAMATH_CALUDE_equation_solution_l2888_288828

theorem equation_solution (x y : ℚ) : 
  19 * (x + y) + 17 = 19 * (-x + y) - 21 ↔ x = -2/19 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2888_288828
