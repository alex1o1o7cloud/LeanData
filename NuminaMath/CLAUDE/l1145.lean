import Mathlib

namespace NUMINAMATH_CALUDE_longer_train_length_l1145_114581

-- Define the given values
def speed_train1 : Real := 60  -- km/hr
def speed_train2 : Real := 40  -- km/hr
def length_shorter : Real := 140  -- meters
def crossing_time : Real := 11.519078473722104  -- seconds

-- Define the theorem
theorem longer_train_length :
  ∃ (length_longer : Real),
    length_longer = 180 ∧
    length_shorter + length_longer =
      (speed_train1 + speed_train2) * 1000 / 3600 * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_longer_train_length_l1145_114581


namespace NUMINAMATH_CALUDE_hospital_nurses_count_l1145_114591

theorem hospital_nurses_count 
  (total_staff : ℕ) 
  (doctor_ratio : ℕ) 
  (nurse_ratio : ℕ) 
  (h1 : total_staff = 200)
  (h2 : doctor_ratio = 4)
  (h3 : nurse_ratio = 6) :
  (nurse_ratio * total_staff) / (doctor_ratio + nurse_ratio) = 120 := by
  sorry

end NUMINAMATH_CALUDE_hospital_nurses_count_l1145_114591


namespace NUMINAMATH_CALUDE_problem_solution_l1145_114525

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (t : ℝ) : Prop :=
  2 * a * t^2 + 12 * t + 9 = 0

-- Define parallel lines
def parallel_lines (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x + b * y = 1 ∧ 4 * x + 18 * y = 3

-- Define the b-th prime number
def nth_prime (b : ℕ) (p : ℕ) : Prop :=
  p.Prime ∧ (Finset.filter Nat.Prime (Finset.range p)).card = b

-- Define the trigonometric equation
def trig_equation (k θ : ℝ) : Prop :=
  k = (4 * Real.sin θ + 3 * Real.cos θ) / (2 * Real.sin θ - Real.cos θ) ∧
  Real.tan θ = 3

theorem problem_solution :
  (∃ a : ℝ, quadratic_equation a has_equal_roots) →
  (∃ b : ℝ, parallel_lines 2 b) →
  (∃ p : ℕ, nth_prime 9 p) →
  (∃ k θ : ℝ, trig_equation k θ) →
  ∃ (a b : ℝ) (p : ℕ) (k : ℝ),
    a = 2 ∧ b = 9 ∧ p = 23 ∧ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1145_114525


namespace NUMINAMATH_CALUDE_tangent_slope_product_l1145_114541

theorem tangent_slope_product (x₀ : ℝ) : 
  let y₁ : ℝ → ℝ := λ x => 2 - 1/x
  let y₂ : ℝ → ℝ := λ x => x^3 - x^2 + x
  let y₁' : ℝ → ℝ := λ x => 1/x^2
  let y₂' : ℝ → ℝ := λ x => 3*x^2 - 2*x + 1
  (y₁' x₀) * (y₂' x₀) = 3 → x₀ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_product_l1145_114541


namespace NUMINAMATH_CALUDE_geometric_sum_proof_l1145_114528

theorem geometric_sum_proof : 
  let a : ℚ := 3/2
  let r : ℚ := 3/2
  let n : ℕ := 15
  let sum : ℚ := (a * (1 - r^n)) / (1 - r)
  sum = 42948417/32768 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_proof_l1145_114528


namespace NUMINAMATH_CALUDE_money_difference_equals_5p_minus_20_l1145_114587

/-- The number of pennies in a nickel -/
def nickel_value : ℕ := 5

/-- The number of nickels Jessica has -/
def jessica_nickels (p : ℕ) : ℕ := 3 * p + 2

/-- The number of nickels Samantha has -/
def samantha_nickels (p : ℕ) : ℕ := 2 * p + 6

/-- The difference in money (in pennies) between Jessica and Samantha -/
def money_difference (p : ℕ) : ℤ :=
  nickel_value * (jessica_nickels p - samantha_nickels p)

theorem money_difference_equals_5p_minus_20 (p : ℕ) :
  money_difference p = 5 * p - 20 := by sorry

end NUMINAMATH_CALUDE_money_difference_equals_5p_minus_20_l1145_114587


namespace NUMINAMATH_CALUDE_no_solution_l1145_114519

theorem no_solution : ¬∃ (A B : ℤ), 
  A = 5 + 3 ∧ 
  B = A - 2 ∧ 
  0 ≤ A ∧ A ≤ 9 ∧ 
  0 ≤ B ∧ B ≤ 9 ∧ 
  0 ≤ A + B ∧ A + B ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l1145_114519


namespace NUMINAMATH_CALUDE_katie_cupcakes_made_l1145_114547

/-- The number of cupcakes Katie made after selling the first batch -/
def cupcakes_made_after (initial sold final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Theorem: Katie made 20 cupcakes after selling the first batch -/
theorem katie_cupcakes_made :
  cupcakes_made_after 26 20 26 = 20 := by
  sorry

end NUMINAMATH_CALUDE_katie_cupcakes_made_l1145_114547


namespace NUMINAMATH_CALUDE_alok_order_l1145_114535

/-- The number of chapatis ordered -/
def chapatis : ℕ := 16

/-- The number of rice plates ordered -/
def rice_plates : ℕ := 5

/-- The number of ice-cream cups ordered -/
def ice_cream_cups : ℕ := 6

/-- The cost of each chapati in rupees -/
def chapati_cost : ℕ := 6

/-- The cost of each rice plate in rupees -/
def rice_cost : ℕ := 45

/-- The cost of each mixed vegetable plate in rupees -/
def veg_cost : ℕ := 70

/-- The total amount paid by Alok in rupees -/
def total_paid : ℕ := 985

/-- The number of mixed vegetable plates ordered by Alok -/
def veg_plates : ℕ := (total_paid - (chapatis * chapati_cost + rice_plates * rice_cost)) / veg_cost

theorem alok_order : veg_plates = 9 := by sorry

end NUMINAMATH_CALUDE_alok_order_l1145_114535


namespace NUMINAMATH_CALUDE_roses_in_vase_correct_l1145_114580

/-- Given a total number of roses and the number of roses left,
    calculate the number of roses put in a vase. -/
def roses_in_vase (total : ℕ) (left : ℕ) : ℕ :=
  total - left

theorem roses_in_vase_correct (total : ℕ) (left : ℕ) 
  (h : left ≤ total) : 
  roses_in_vase total left = total - left :=
by
  sorry

#eval roses_in_vase 29 12  -- Should evaluate to 17

end NUMINAMATH_CALUDE_roses_in_vase_correct_l1145_114580


namespace NUMINAMATH_CALUDE_keanu_fish_problem_l1145_114536

theorem keanu_fish_problem :
  ∀ (dog_fish cat_fish : ℕ),
    cat_fish = dog_fish / 2 →
    dog_fish + cat_fish = 240 / 4 →
    dog_fish = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_keanu_fish_problem_l1145_114536


namespace NUMINAMATH_CALUDE_fifth_observation_value_l1145_114575

theorem fifth_observation_value (x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℝ) :
  (x1 + x2 + x3 + x4 + x5) / 5 = 10 →
  (x5 + x6 + x7 + x8 + x9) / 5 = 8 →
  (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) / 9 = 8 →
  x5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_fifth_observation_value_l1145_114575


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1145_114574

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1145_114574


namespace NUMINAMATH_CALUDE_triangle_third_side_minimum_l1145_114595

theorem triangle_third_side_minimum (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a - b = 5 ∨ b - a = 5) →
  Even (a + b + c) →
  c ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_minimum_l1145_114595


namespace NUMINAMATH_CALUDE_yarn_balls_per_sweater_l1145_114509

/-- The number of balls of yarn needed for each sweater -/
def balls_per_sweater : ℕ := sorry

/-- The cost of each ball of yarn in dollars -/
def yarn_cost : ℕ := 6

/-- The selling price of each sweater in dollars -/
def sweater_price : ℕ := 35

/-- The number of sweaters sold -/
def sweaters_sold : ℕ := 28

/-- The total profit from selling all sweaters in dollars -/
def total_profit : ℕ := 308

theorem yarn_balls_per_sweater :
  (sweaters_sold * (sweater_price - yarn_cost * balls_per_sweater) = total_profit) →
  balls_per_sweater = 4 := by sorry

end NUMINAMATH_CALUDE_yarn_balls_per_sweater_l1145_114509


namespace NUMINAMATH_CALUDE_total_shared_amount_l1145_114516

theorem total_shared_amount 
  (T a b c d : ℚ) 
  (h1 : a = (1/3) * (b + c + d))
  (h2 : b = (2/7) * (a + c + d))
  (h3 : c = (4/9) * (a + b + d))
  (h4 : d = (5/11) * (a + b + c))
  (h5 : a = b + 20)
  (h6 : c = d - 15)
  (h7 : T = a + b + c + d)
  (h8 : ∃ k : ℤ, T = 10 * k) :
  T = 1330 := by
sorry

end NUMINAMATH_CALUDE_total_shared_amount_l1145_114516


namespace NUMINAMATH_CALUDE_tooth_arrangements_count_l1145_114501

/-- The number of unique arrangements of letters in TOOTH -/
def toothArrangements : ℕ :=
  Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of letters in TOOTH is 10 -/
theorem tooth_arrangements_count : toothArrangements = 10 := by
  sorry

end NUMINAMATH_CALUDE_tooth_arrangements_count_l1145_114501


namespace NUMINAMATH_CALUDE_speed_ratio_proof_l1145_114589

/-- Proves that the speed ratio of return to outbound trip is 2:1 given specific conditions -/
theorem speed_ratio_proof (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) : 
  total_distance = 40 ∧ 
  total_time = 6 ∧ 
  return_speed = 10 → 
  (return_speed / (total_distance / 2 / (total_time - total_distance / 2 / return_speed))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_proof_l1145_114589


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_is_even_l1145_114569

theorem n_times_n_plus_one_is_even (n : ℤ) : 2 ∣ n * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_is_even_l1145_114569


namespace NUMINAMATH_CALUDE_expression_simplification_l1145_114524

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x^2 - 1) / (x^2 + x) / (x - (2*x - 1) / x) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1145_114524


namespace NUMINAMATH_CALUDE_burning_time_3x5_grid_l1145_114515

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  rows : ℕ
  cols : ℕ
  toothpicks : ℕ

/-- Represents the burning properties of toothpicks -/
structure BurningProperties where
  burn_time : ℕ  -- Time for one toothpick to burn completely
  spread_speed : ℝ  -- Speed at which fire spreads (assumed constant)

/-- Calculates the maximum burning time for a toothpick grid -/
def max_burning_time (grid : ToothpickGrid) (props : BurningProperties) : ℕ :=
  sorry  -- The actual calculation would go here

/-- Theorem stating the maximum burning time for the specific grid -/
theorem burning_time_3x5_grid :
  let grid := ToothpickGrid.mk 3 5 38
  let props := BurningProperties.mk 10 1
  max_burning_time grid props = 65 :=
sorry

#check burning_time_3x5_grid

end NUMINAMATH_CALUDE_burning_time_3x5_grid_l1145_114515


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l1145_114514

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to a natural number -/
def from_binary (l : List Bool) : ℕ :=
  l.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem base_2_representation_of_123 :
  to_binary 123 = [true, true, true, true, false, true, true] :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l1145_114514


namespace NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l1145_114566

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_18gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
    (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by sorry

end NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l1145_114566


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_and_eccentricity_l1145_114518

/-- The focal length of a hyperbola with equation x^2 - y^2/3 = 1 is 4 and its eccentricity is 2 -/
theorem hyperbola_focal_length_and_eccentricity :
  let a : ℝ := 1
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let focal_length : ℝ := 2 * c
  let eccentricity : ℝ := c / a
  focal_length = 4 ∧ eccentricity = 2 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_focal_length_and_eccentricity_l1145_114518


namespace NUMINAMATH_CALUDE_simplified_expression_implies_A_l1145_114583

/-- 
Given that (A - 3 / (a - 1)) * ((2 * a - 2) / (a + 2)) = 2 * a - 4,
prove that A = a + 1
-/
theorem simplified_expression_implies_A (a : ℝ) (A : ℝ) 
  (h : (A - 3 / (a - 1)) * ((2 * a - 2) / (a + 2)) = 2 * a - 4) :
  A = a + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_implies_A_l1145_114583


namespace NUMINAMATH_CALUDE_digit_placement_combinations_l1145_114522

def grid_size : ℕ := 6
def num_digits : ℕ := 4

theorem digit_placement_combinations : 
  (grid_size * (grid_size - 1) * (grid_size - 2) * (grid_size - 3) * (grid_size - 4)) = 720 :=
by sorry

end NUMINAMATH_CALUDE_digit_placement_combinations_l1145_114522


namespace NUMINAMATH_CALUDE_max_correct_answers_l1145_114506

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (wrong_points : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_points = 5 →
  wrong_points = -2 →
  total_score = 150 →
  (∃ (correct unpicked wrong : ℕ),
    correct + unpicked + wrong = total_questions ∧
    correct * correct_points + wrong * wrong_points = total_score) →
  (∀ (x : ℕ), x > 38 →
    ¬∃ (unpicked wrong : ℕ),
      x + unpicked + wrong = total_questions ∧
      x * correct_points + wrong * wrong_points = total_score) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1145_114506


namespace NUMINAMATH_CALUDE_cosine_power_identity_l1145_114567

theorem cosine_power_identity (θ : ℝ) (u : ℝ) (n : ℤ) :
  2 * Real.cos θ = u + 1 / u →
  2 * Real.cos (n * θ) = u^n + 1 / u^n :=
by sorry

end NUMINAMATH_CALUDE_cosine_power_identity_l1145_114567


namespace NUMINAMATH_CALUDE_special_polynomial_derivative_theorem_l1145_114500

/-- A second-degree polynomial with roots in [-1, 1] and |f(x₀)| = 1 for some x₀ ∈ [-1, 1] -/
structure SpecialPolynomial where
  f : ℝ → ℝ
  degree_two : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  roots_in_interval : ∀ r, f r = 0 → r ∈ Set.Icc (-1 : ℝ) 1
  exists_unit_value : ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |f x₀| = 1

/-- The main theorem about special polynomials -/
theorem special_polynomial_derivative_theorem (p : SpecialPolynomial) :
  (∀ α ∈ Set.Icc (0 : ℝ) 1, ∃ ζ ∈ Set.Icc (-1 : ℝ) 1, |deriv p.f ζ| = α) ∧
  (¬∃ ζ ∈ Set.Icc (-1 : ℝ) 1, |deriv p.f ζ| > 1) :=
by sorry

end NUMINAMATH_CALUDE_special_polynomial_derivative_theorem_l1145_114500


namespace NUMINAMATH_CALUDE_supplementary_angles_difference_l1145_114558

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  (∃ k : ℕ, a = 15 * k ∨ b = 15 * k) →  -- one angle is multiple of 15
  |a - b| = 45 := by
sorry

end NUMINAMATH_CALUDE_supplementary_angles_difference_l1145_114558


namespace NUMINAMATH_CALUDE_circle_radii_equation_l1145_114593

theorem circle_radii_equation (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (ea : a = 1 / a) (eb : b = 1 / b) (ec : c = 1 / c) (ed : d = 1 / d) :
  2 * (a^2 + b^2 + c^2 + d^2) = (a + b + c + d)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_equation_l1145_114593


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1145_114520

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ),
    (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
      (x^2 - 5*x + 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) ∧
    A = -6 ∧ B = 7 ∧ C = -5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1145_114520


namespace NUMINAMATH_CALUDE_perpendicular_line_exists_l1145_114521

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define perpendicularity
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define a point being on a line
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

-- Define a point being on a circle
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define a line passing through a point
def line_through_point (l : Line) (p : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem perpendicular_line_exists 
  (C : Circle) (A B M : ℝ × ℝ) (diameter : Line) :
  point_on_circle A C →
  point_on_circle B C →
  point_on_line A diameter →
  point_on_line B diameter →
  ∃ (L : Line), line_through_point L M ∧ perpendicular L diameter :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_exists_l1145_114521


namespace NUMINAMATH_CALUDE_sum_a_b_equals_one_l1145_114533

theorem sum_a_b_equals_one (a b : ℝ) (h : a = Real.sqrt (2 * b - 4) + Real.sqrt (4 - 2 * b) - 1) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_one_l1145_114533


namespace NUMINAMATH_CALUDE_circle_tangent_to_semicircles_radius_bounds_l1145_114556

/-- Given a triangle ABC with semiperimeter s and inradius r, and semicircles drawn on its sides,
    the radius t of the circle tangent to all three semicircles satisfies:
    s/2 < t ≤ s/2 + (1 - √3/2)r -/
theorem circle_tangent_to_semicircles_radius_bounds
  (s r t : ℝ) -- semiperimeter, inradius, and radius of tangent circle
  (h_s_pos : 0 < s)
  (h_r_pos : 0 < r)
  (h_t_pos : 0 < t)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ s = (a + b + c) / 2)
  (h_inradius : ∃ (area : ℝ), area > 0 ∧ r = area / s)
  (h_tangent : ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
               t + x/2 = t + y/2 ∧ t + y/2 = t + z/2 ∧ x + y + z = 2 * s) :
  s / 2 < t ∧ t ≤ s / 2 + (1 - Real.sqrt 3 / 2) * r := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_semicircles_radius_bounds_l1145_114556


namespace NUMINAMATH_CALUDE_inscribed_triangle_existence_l1145_114590

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle defined by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a triangle is inscribed in a circle -/
def isInscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

/-- Calculate an angle of a triangle -/
def angle (t : Triangle) (vertex : ℝ × ℝ) : ℝ :=
  sorry

/-- Calculate the length of a median in a triangle -/
def medianLength (t : Triangle) (vertex : ℝ × ℝ) : ℝ :=
  sorry

/-- The main theorem -/
theorem inscribed_triangle_existence (k : Circle) (α : ℝ) (s_b : ℝ) :
  ∃ n : Fin 3, ∃ triangles : Fin n → Triangle,
    (∀ i, isInscribed (triangles i) k) ∧
    (∀ i, ∃ v, angle (triangles i) v = α) ∧
    (∀ i, ∃ v, medianLength (triangles i) v = s_b) :=
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_existence_l1145_114590


namespace NUMINAMATH_CALUDE_mrs_randall_third_grade_years_l1145_114552

/-- Represents the number of years Mrs. Randall has been teaching -/
def total_teaching_years : ℕ := 26

/-- Represents the number of years Mrs. Randall taught second grade -/
def second_grade_years : ℕ := 8

/-- Represents the number of years Mrs. Randall has taught third grade -/
def third_grade_years : ℕ := total_teaching_years - second_grade_years

theorem mrs_randall_third_grade_years :
  third_grade_years = 18 :=
by sorry

end NUMINAMATH_CALUDE_mrs_randall_third_grade_years_l1145_114552


namespace NUMINAMATH_CALUDE_collinear_probability_l1145_114578

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The number of dots to be chosen -/
def dotsChosen : ℕ := 4

/-- The total number of ways to choose 4 dots from a 5x5 grid -/
def totalWays : ℕ := Nat.choose (gridSize * gridSize) dotsChosen

/-- The number of ways to choose 4 collinear dots -/
def collinearWays : ℕ := 
  gridSize * Nat.choose gridSize dotsChosen + -- Horizontal lines
  gridSize * Nat.choose gridSize dotsChosen + -- Vertical lines
  2 * Nat.choose gridSize dotsChosen +        -- Main diagonals
  4                                           -- Adjacent diagonals

/-- The probability of choosing 4 collinear dots from a 5x5 grid -/
theorem collinear_probability : 
  (collinearWays : ℚ) / totalWays = 64 / 12650 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_l1145_114578


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1145_114555

/-- Given a line in vector form, prove it's equivalent to slope-intercept form --/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y + 4) = 0 ↔ y = 2 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1145_114555


namespace NUMINAMATH_CALUDE_derivative_f_zero_dne_l1145_114532

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 6 * x + x * Real.sin (1 / x) else 0

theorem derivative_f_zero_dne :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| → |h| < δ →
    |((f (0 + h) - f 0) / h) - L| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_f_zero_dne_l1145_114532


namespace NUMINAMATH_CALUDE_fourth_term_is_one_l1145_114511

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℚ, a (n + 1) = a n * q
  first_fifth_diff : a 1 - a 5 = -15/2
  sum_first_four : (a 1) + (a 2) + (a 3) + (a 4) = -5

/-- The fourth term of the geometric sequence is 1 -/
theorem fourth_term_is_one (seq : GeometricSequence) : seq.a 4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_fourth_term_is_one_l1145_114511


namespace NUMINAMATH_CALUDE_unique_abc_solution_l1145_114543

/-- Represents a base-7 number with two digits -/
def Base7TwoDigit (a b : Nat) : Nat := 7 * a + b

/-- Represents a base-7 number with one digit -/
def Base7OneDigit (c : Nat) : Nat := c

theorem unique_abc_solution :
  ∀ A B C : Nat,
    A < 7 → B < 7 → C < 7 →
    A ≠ B → B ≠ C → A ≠ C →
    Base7TwoDigit A B + Base7OneDigit C = Base7TwoDigit C 0 →
    Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit C C →
    A = 5 ∧ B = 1 ∧ C = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_abc_solution_l1145_114543


namespace NUMINAMATH_CALUDE_soda_barrel_leak_time_l1145_114576

/-- The time it takes to fill one barrel with the leak -/
def leak_fill_time : ℝ := 5

/-- The normal filling time for one barrel -/
def normal_fill_time : ℝ := 3

/-- The number of barrels -/
def num_barrels : ℝ := 12

/-- The additional time it takes to fill all barrels with the leak -/
def additional_time : ℝ := 24

theorem soda_barrel_leak_time :
  leak_fill_time * num_barrels = normal_fill_time * num_barrels + additional_time :=
by sorry

end NUMINAMATH_CALUDE_soda_barrel_leak_time_l1145_114576


namespace NUMINAMATH_CALUDE_cyrus_pages_left_l1145_114530

/-- Represents the number of pages Cyrus writes on each day --/
def pages_written : Fin 4 → ℕ
| 0 => 25  -- Day 1
| 1 => 2 * 25  -- Day 2
| 2 => 2 * (2 * 25)  -- Day 3
| 3 => 10  -- Day 4

/-- The total number of pages Cyrus needs to write --/
def total_pages : ℕ := 500

/-- The number of pages Cyrus still needs to write --/
def pages_left : ℕ := total_pages - (pages_written 0 + pages_written 1 + pages_written 2 + pages_written 3)

theorem cyrus_pages_left : pages_left = 315 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_pages_left_l1145_114530


namespace NUMINAMATH_CALUDE_tony_initial_money_l1145_114513

/-- Given Tony's expenses and remaining money, prove his initial amount --/
theorem tony_initial_money :
  ∀ (initial spent_ticket spent_hotdog remaining : ℕ),
    spent_ticket = 8 →
    spent_hotdog = 3 →
    remaining = 9 →
    initial = spent_ticket + spent_hotdog + remaining →
    initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_tony_initial_money_l1145_114513


namespace NUMINAMATH_CALUDE_amanda_earnings_l1145_114582

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Total hours worked on Monday -/
def monday_hours : ℝ := 5 * 1.5

/-- Total hours worked on Tuesday -/
def tuesday_hours : ℝ := 3

/-- Total hours worked on Thursday -/
def thursday_hours : ℝ := 2 * 2

/-- Total hours worked on Saturday -/
def saturday_hours : ℝ := 6

/-- Total hours worked in the week -/
def total_hours : ℝ := monday_hours + tuesday_hours + thursday_hours + saturday_hours

/-- Amanda's total earnings for the week -/
def total_earnings : ℝ := hourly_rate * total_hours

theorem amanda_earnings : total_earnings = 410 := by
  sorry

end NUMINAMATH_CALUDE_amanda_earnings_l1145_114582


namespace NUMINAMATH_CALUDE_smallest_positive_w_l1145_114546

theorem smallest_positive_w (y w : Real) (h1 : Real.sin y = 0) (h2 : Real.sin (y + w) = Real.sqrt 3 / 2) :
  ∃ (w_min : Real), w_min > 0 ∧ w_min = π / 3 ∧ ∀ (w' : Real), w' > 0 ∧ Real.sin y = 0 ∧ Real.sin (y + w') = Real.sqrt 3 / 2 → w' ≥ w_min :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_w_l1145_114546


namespace NUMINAMATH_CALUDE_kat_boxing_hours_l1145_114598

/-- Represents Kat's weekly training schedule -/
structure TrainingSchedule where
  strength_sessions : ℕ
  strength_hours_per_session : ℚ
  boxing_sessions : ℕ
  total_hours : ℚ

/-- Calculates the number of hours Kat trains at the boxing gym each time -/
def boxing_hours_per_session (schedule : TrainingSchedule) : ℚ :=
  (schedule.total_hours - schedule.strength_sessions * schedule.strength_hours_per_session) / schedule.boxing_sessions

/-- Theorem stating that Kat trains 1.5 hours at the boxing gym each time -/
theorem kat_boxing_hours (schedule : TrainingSchedule) 
  (h1 : schedule.strength_sessions = 3)
  (h2 : schedule.strength_hours_per_session = 1)
  (h3 : schedule.boxing_sessions = 4)
  (h4 : schedule.total_hours = 9) :
  boxing_hours_per_session schedule = 3/2 := by
  sorry

#eval boxing_hours_per_session { strength_sessions := 3, strength_hours_per_session := 1, boxing_sessions := 4, total_hours := 9 }

end NUMINAMATH_CALUDE_kat_boxing_hours_l1145_114598


namespace NUMINAMATH_CALUDE_exists_function_satisfying_equation_l1145_114549

theorem exists_function_satisfying_equation : 
  ∃ f : ℤ → ℤ, ∀ a b : ℤ, f (a + b) - f (a * b) = f a * f b - 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_equation_l1145_114549


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1145_114545

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : is_geometric_sequence a)
  (h_prod1 : a 1 * a 2 * a 3 = 4)
  (h_prod2 : a 4 * a 5 * a 6 = 12)
  (h_prod3 : ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324) :
  ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324 ∧ n = 14 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1145_114545


namespace NUMINAMATH_CALUDE_line_plane_perp_sufficiency_not_necessity_l1145_114565

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicularity relation between a line and a plane
variable (line_perp_plane : Line → Plane → Prop)

-- Define the perpendicularity relation between two planes
variable (plane_perp_plane : Plane → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_perp_sufficiency_not_necessity
  (α β : Plane) (m : Line)
  (h_diff : α ≠ β)
  (h_m_in_α : line_in_plane m α) :
  (line_perp_plane m β → plane_perp_plane α β) ∧
  ¬(plane_perp_plane α β → line_perp_plane m β) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perp_sufficiency_not_necessity_l1145_114565


namespace NUMINAMATH_CALUDE_min_sum_with_condition_min_sum_equality_l1145_114554

theorem min_sum_with_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  a + b ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  a + b = 3 + 2 * Real.sqrt 2 ↔ a = 1 + Real.sqrt 2 ∧ b = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_condition_min_sum_equality_l1145_114554


namespace NUMINAMATH_CALUDE_nested_square_root_twenty_l1145_114563

theorem nested_square_root_twenty : 
  ∃ x : ℝ, x = Real.sqrt (20 + x) ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_nested_square_root_twenty_l1145_114563


namespace NUMINAMATH_CALUDE_beta_value_l1145_114531

theorem beta_value (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_α_β : Real.sin (α - β) = -(Real.sqrt 10) / 10) : β = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_beta_value_l1145_114531


namespace NUMINAMATH_CALUDE_odd_function_sum_l1145_114544

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1145_114544


namespace NUMINAMATH_CALUDE_triangle_max_area_l1145_114577

/-- Given a triangle ABC where AB = 9 and BC:AC = 3:4, 
    the maximum possible area of the triangle is 243 / (2√7) square units. -/
theorem triangle_max_area (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  AB = 9 ∧ BC / AC = 3 / 4 → 
  area ≤ 243 / (2 * Real.sqrt 7) := by
sorry


end NUMINAMATH_CALUDE_triangle_max_area_l1145_114577


namespace NUMINAMATH_CALUDE_set_operations_l1145_114559

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {0, 1, 4}

-- Define set B
def B : Finset Nat := {0, 1, 3}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {0, 1}) ∧ (A ∪ B = {0, 1, 3, 4}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1145_114559


namespace NUMINAMATH_CALUDE_set_operations_l1145_114510

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 0 ≤ x ∧ x < 5}

def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

theorem set_operations :
  (A ∩ B = {x | 0 ≤ x ∧ x < 4}) ∧
  (A ∪ B = {x | -2 ≤ x ∧ x < 5}) ∧
  (A ∩ (U \ B) = {x | 4 ≤ x ∧ x < 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1145_114510


namespace NUMINAMATH_CALUDE_find_y_l1145_114539

theorem find_y : ∃ y : ℝ, y > 0 ∧ 16 * y = 256 ∧ ∃ n : ℕ, y^2 = n ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l1145_114539


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1145_114542

-- Define the points
def M : ℝ × ℝ := (-2, 3)
def P : ℝ × ℝ := (1, 0)

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the Law of Reflection
def law_of_reflection (incident : ℝ × ℝ → ℝ × ℝ → Prop) (reflected : ℝ × ℝ → ℝ × ℝ → Prop) : Prop :=
  ∀ p q r, incident p q → reflected q r → (q.2 = 0) → 
    (p.2 - q.2) * (r.1 - q.1) = (r.2 - q.2) * (p.1 - q.1)

-- State the theorem
theorem reflected_ray_equation :
  ∃ (incident reflected : ℝ × ℝ → ℝ × ℝ → Prop),
    incident M P ∧ P ∈ x_axis ∧ law_of_reflection incident reflected →
    ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧
               ∀ x y : ℝ, reflected P (x, y) ↔ a * x + b * y + c = 0 ∧
               a = 1 ∧ b = 1 ∧ c = -1 := by
  sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1145_114542


namespace NUMINAMATH_CALUDE_juan_oranges_picked_l1145_114538

def total_oranges : ℕ := 107
def del_daily_pick : ℕ := 23
def del_days : ℕ := 2

theorem juan_oranges_picked : 
  total_oranges - (del_daily_pick * del_days) = 61 := by
  sorry

end NUMINAMATH_CALUDE_juan_oranges_picked_l1145_114538


namespace NUMINAMATH_CALUDE_teachers_arrangement_count_l1145_114588

def number_of_seats : ℕ := 25
def number_of_teachers : ℕ := 5
def min_gap : ℕ := 2

def arrange_teachers (seats : ℕ) (teachers : ℕ) (gap : ℕ) : ℕ :=
  Nat.choose (seats + teachers - (teachers - 1) * (gap + 1)) teachers

theorem teachers_arrangement_count :
  arrange_teachers number_of_seats number_of_teachers min_gap = 26334 := by
  sorry

end NUMINAMATH_CALUDE_teachers_arrangement_count_l1145_114588


namespace NUMINAMATH_CALUDE_count_pairs_eq_45_l1145_114584

def count_pairs : Nat :=
  (Finset.range 6).sum fun m =>
    (Finset.range ((40 - (m + 1)^2) / 3 + 1)).card

theorem count_pairs_eq_45 : count_pairs = 45 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_45_l1145_114584


namespace NUMINAMATH_CALUDE_ab_value_l1145_114548

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 33) : a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1145_114548


namespace NUMINAMATH_CALUDE_r_plus_s_value_l1145_114503

/-- The line equation y = -5/3 * x + 15 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is where the line crosses the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is where the line crosses the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- T is a point on line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- Area of triangle POQ is 4 times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 * Q.2 - Q.1 * P.2) / 2) = 4 * abs ((P.1 * s - r * P.2) / 2)

/-- Main theorem: Given the conditions, r + s = 10.5 -/
theorem r_plus_s_value (r s : ℝ) 
  (h1 : line_equation r s) 
  (h2 : T_on_PQ r s) 
  (h3 : area_condition r s) : 
  r + s = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_r_plus_s_value_l1145_114503


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l1145_114573

/-- The volume of a cylinder formed by rotating a rectangle about its vertical line of symmetry -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (h_length : length = 20) (h_width : width = 10) :
  let radius := width / 2
  let height := length
  let volume := π * radius^2 * height
  volume = 500 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l1145_114573


namespace NUMINAMATH_CALUDE_pyramid_inscribed_cube_volume_l1145_114553

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  height : ℝ

/-- A cube inscribed in the pyramid -/
structure InscribedCube where
  edge_length : ℝ

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.edge_length ^ 3

theorem pyramid_inscribed_cube_volume 
  (p : Pyramid) 
  (c : InscribedCube) 
  (h_base : p.base_side = 2) 
  (h_height : p.height = Real.sqrt 6) 
  (h_cube_edge : c.edge_length = Real.sqrt 6 / 3) : 
  cube_volume c = 2 * Real.sqrt 6 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_inscribed_cube_volume_l1145_114553


namespace NUMINAMATH_CALUDE_multiples_of_15_between_10_and_150_l1145_114527

theorem multiples_of_15_between_10_and_150 : 
  ∃ n : ℕ, n = (Finset.filter (λ x => 15 ∣ x ∧ x > 10 ∧ x < 150) (Finset.range 150)).card ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_10_and_150_l1145_114527


namespace NUMINAMATH_CALUDE_palindrome_count_l1145_114572

/-- Represents a time on a 12-hour digital clock --/
structure Time where
  hour : Nat
  minute : Nat
  hour_valid : 1 ≤ hour ∧ hour ≤ 12
  minute_valid : minute < 60

/-- Checks if a given time is a palindrome --/
def isPalindrome (t : Time) : Bool :=
  let digits := 
    if t.hour < 10 then
      [t.hour, t.minute / 10, t.minute % 10]
    else
      [t.hour / 10, t.hour % 10, t.minute / 10, t.minute % 10]
  digits = digits.reverse

/-- The set of all valid palindrome times on a 12-hour digital clock --/
def palindromeTimes : Finset Time :=
  sorry

theorem palindrome_count : palindromeTimes.card = 57 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_count_l1145_114572


namespace NUMINAMATH_CALUDE_base_5_minus_base_7_digits_l1145_114537

-- Define the number of digits in a given base
def numDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

-- State the theorem
theorem base_5_minus_base_7_digits : 
  numDigits 2023 5 - numDigits 2023 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_5_minus_base_7_digits_l1145_114537


namespace NUMINAMATH_CALUDE_count_negative_numbers_l1145_114534

def given_set : Finset Int := {-3, -2, 0, 5}

theorem count_negative_numbers : 
  (given_set.filter (λ x => x < 0)).card = 2 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l1145_114534


namespace NUMINAMATH_CALUDE_canteen_banana_requirement_l1145_114586

/-- The number of bananas required for the given period -/
def total_bananas : ℕ := 9828

/-- The number of weeks in the given period -/
def num_weeks : ℕ := 9

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of bananas in a dozen -/
def bananas_per_dozen : ℕ := 12

/-- Theorem: The canteen needs 13 dozens of bananas per day -/
theorem canteen_banana_requirement :
  (total_bananas / (num_weeks * days_per_week)) / bananas_per_dozen = 13 := by
  sorry

end NUMINAMATH_CALUDE_canteen_banana_requirement_l1145_114586


namespace NUMINAMATH_CALUDE_apple_sale_loss_l1145_114568

/-- The fraction of the cost price lost by a seller when selling an item -/
def fractionLost (sellingPrice costPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

/-- Theorem: The fraction of the cost price lost when selling an apple for 19 Rs with a cost price of 20 Rs is 1/20 -/
theorem apple_sale_loss : fractionLost 19 20 = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_apple_sale_loss_l1145_114568


namespace NUMINAMATH_CALUDE_fifth_term_is_648_l1145_114561

/-- A geometric sequence with 7 terms, first term 8, and last term 5832 -/
def GeometricSequence : Type := 
  { a : Fin 7 → ℝ // a 0 = 8 ∧ a 6 = 5832 ∧ ∀ i j, i < j → (a j) / (a i) = (a 1) / (a 0) }

/-- The fifth term of the geometric sequence is 648 -/
theorem fifth_term_is_648 (seq : GeometricSequence) : seq.val 4 = 648 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_648_l1145_114561


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1145_114557

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}
def B : Set ℝ := {x | (x-3)*(2-x) ≥ 0}

-- State the theorem
theorem necessary_not_sufficient_condition (a : ℝ) (h1 : a > 0) 
  (h2 : B ⊂ A a) (h3 : A a ≠ B) : a ∈ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1145_114557


namespace NUMINAMATH_CALUDE_a_squared_lt_one_sufficient_not_necessary_for_a_lt_two_l1145_114592

theorem a_squared_lt_one_sufficient_not_necessary_for_a_lt_two :
  (∀ a : ℝ, a^2 < 1 → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_a_squared_lt_one_sufficient_not_necessary_for_a_lt_two_l1145_114592


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l1145_114571

/-- The standard equation of an ellipse given its focal distance and sum of distances from a point to foci -/
theorem ellipse_standard_equation (focal_distance sum_distances : ℝ) :
  focal_distance = 8 →
  sum_distances = 10 →
  (∃ x y : ℝ, x^2 / 25 + y^2 / 9 = 1) ∨ (∃ x y : ℝ, x^2 / 9 + y^2 / 25 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l1145_114571


namespace NUMINAMATH_CALUDE_green_leaves_remaining_l1145_114570

theorem green_leaves_remaining (num_plants : ℕ) (initial_leaves : ℕ) (falling_fraction : ℚ) : 
  num_plants = 3 → 
  initial_leaves = 18 → 
  falling_fraction = 1/3 → 
  (num_plants * initial_leaves * (1 - falling_fraction) : ℚ) = 36 := by
sorry

end NUMINAMATH_CALUDE_green_leaves_remaining_l1145_114570


namespace NUMINAMATH_CALUDE_lattice_point_proximity_probability_l1145_114597

theorem lattice_point_proximity_probability (d : ℝ) : 
  (d > 0) → 
  (π * d^2 = 1/3) → 
  (d = Real.sqrt (1 / (3 * π))) :=
by sorry

end NUMINAMATH_CALUDE_lattice_point_proximity_probability_l1145_114597


namespace NUMINAMATH_CALUDE_intersection_sum_l1145_114508

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 1
def g (x y : ℝ) : Prop := x + 3*y = 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | f p.1 = p.2 ∧ g p.1 p.2}

-- State the theorem
theorem intersection_sum :
  ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    p₁ ∈ intersection_points ∧
    p₂ ∈ intersection_points ∧
    p₃ ∈ intersection_points ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    p₁.1 + p₂.1 + p₃.1 = 3 ∧
    p₁.2 + p₂.2 + p₃.2 = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1145_114508


namespace NUMINAMATH_CALUDE_fifth_term_is_1280_l1145_114502

/-- A geometric sequence of positive integers with first term 5 and fourth term 320 -/
def geometric_sequence (n : ℕ) : ℕ :=
  5 * (320 / 5) ^ ((n - 1) / 3)

/-- The fifth term of the geometric sequence is 1280 -/
theorem fifth_term_is_1280 : geometric_sequence 5 = 1280 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_1280_l1145_114502


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1145_114551

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 55,
    where one side of the equilateral triangle is a side of the isosceles triangle,
    the base of the isosceles triangle is 15 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 60)
  (h_isosceles_perimeter : isosceles_perimeter = 55)
  (h_shared_side : equilateral_perimeter / 3 = isosceles_perimeter / 2 - isosceles_base / 2) :
  isosceles_base = 15 :=
by
  sorry

#check isosceles_triangle_base_length

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1145_114551


namespace NUMINAMATH_CALUDE_problem_solution_l1145_114512

theorem problem_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + a*b + b^2 = 9)
  (h2 : b^2 + b*c + c^2 = 52)
  (h3 : c^2 + c*a + a^2 = 49) :
  (49*b^2 - 33*b*c + 9*c^2) / a^2 = 52 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1145_114512


namespace NUMINAMATH_CALUDE_calculate_tip_percentage_l1145_114505

/-- Calculates the percentage tip given the prices of four ice cream sundaes and the final bill -/
theorem calculate_tip_percentage (price1 price2 price3 price4 final_bill : ℚ) : 
  price1 = 9 ∧ price2 = 7.5 ∧ price3 = 10 ∧ price4 = 8.5 ∧ final_bill = 42 →
  (final_bill - (price1 + price2 + price3 + price4)) / (price1 + price2 + price3 + price4) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_tip_percentage_l1145_114505


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1145_114504

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = -p.1 + 1}
def N : Set (ℝ × ℝ) := {p | p.2 = p.1 - 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(1, 0)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1145_114504


namespace NUMINAMATH_CALUDE_divisibility_problem_l1145_114523

theorem divisibility_problem (A B C : Nat) : 
  A < 10 → B < 10 → C < 10 →
  (7 * 1000000 + A * 100000 + 5 * 10000 + 1 * 1000 + B * 10 + 2) % 15 = 0 →
  (3 * 1000000 + 2 * 100000 + 6 * 10000 + A * 1000 + B * 100 + 4 * 10 + C) % 15 = 0 →
  C = 4 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1145_114523


namespace NUMINAMATH_CALUDE_discount_rates_sum_l1145_114529

-- Define the regular prices
def fox_price : ℝ := 15
def pony_price : ℝ := 18

-- Define the number of pairs purchased
def fox_pairs : ℕ := 3
def pony_pairs : ℕ := 2

-- Define the total savings
def total_savings : ℝ := 8.55

-- Define the approximate discount rate for Pony jeans
def pony_discount_rate : ℝ := 0.15

-- Define the discount rates as variables
variable (fox_discount_rate : ℝ)

-- Theorem statement
theorem discount_rates_sum :
  fox_discount_rate + pony_discount_rate = 0.22 :=
by sorry

end NUMINAMATH_CALUDE_discount_rates_sum_l1145_114529


namespace NUMINAMATH_CALUDE_equality_except_two_l1145_114585

theorem equality_except_two (x : ℝ) : 
  x ≠ 2 → (x^2 - 4*x + 4) / (x - 2) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_equality_except_two_l1145_114585


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l1145_114594

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 2 * y

-- State the theorem
theorem heartsuit_three_eight : heartsuit 3 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l1145_114594


namespace NUMINAMATH_CALUDE_laura_change_l1145_114550

/-- Calculates the change Laura should receive after her shopping trip -/
theorem laura_change : 
  let pants_cost : ℕ := 2 * 64
  let shirts_cost : ℕ := 4 * 42
  let shoes_cost : ℕ := 3 * 78
  let jackets_cost : ℕ := 2 * 103
  let watch_cost : ℕ := 215
  let jewelry_cost : ℕ := 2 * 120
  let total_cost : ℕ := pants_cost + shirts_cost + shoes_cost + jackets_cost + watch_cost + jewelry_cost
  let amount_given : ℕ := 800
  Int.ofNat amount_given - Int.ofNat total_cost = -391 := by
  sorry

end NUMINAMATH_CALUDE_laura_change_l1145_114550


namespace NUMINAMATH_CALUDE_accounting_client_time_ratio_l1145_114526

/-- Given a total work time and time spent calling clients, 
    calculate the ratio of time spent doing accounting to time spent calling clients. -/
theorem accounting_client_time_ratio 
  (total_time : ℕ) 
  (client_time : ℕ) 
  (h1 : total_time = 560) 
  (h2 : client_time = 70) : 
  (total_time - client_time) / client_time = 7 := by
  sorry

end NUMINAMATH_CALUDE_accounting_client_time_ratio_l1145_114526


namespace NUMINAMATH_CALUDE_amount_to_fifth_sixth_homes_l1145_114599

/-- The amount donated to the fifth and sixth nursing homes combined -/
def amount_fifth_sixth (total donation_1 donation_2 donation_3 donation_4 : ℕ) : ℕ :=
  total - (donation_1 + donation_2 + donation_3 + donation_4)

/-- Theorem stating the amount given to the fifth and sixth nursing homes -/
theorem amount_to_fifth_sixth_homes :
  amount_fifth_sixth 10000 2750 1945 1275 1890 = 2140 := by
  sorry

end NUMINAMATH_CALUDE_amount_to_fifth_sixth_homes_l1145_114599


namespace NUMINAMATH_CALUDE_possible_m_values_l1145_114560

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- Define the theorem
theorem possible_m_values :
  ∀ m : ℝ, (B m ⊆ A m) → (m = 0 ∨ m = 3) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_possible_m_values_l1145_114560


namespace NUMINAMATH_CALUDE_min_prime_no_solution_l1145_114517

theorem min_prime_no_solution : 
  ∀ p : ℕ, Prime p → p > 3 →
    (∀ n : ℕ, n > 0 → ¬(2^n + 3^n) % p = 0) →
    p ≥ 19 :=
by sorry

end NUMINAMATH_CALUDE_min_prime_no_solution_l1145_114517


namespace NUMINAMATH_CALUDE_no_prime_in_first_15_cumulative_sums_l1145_114507

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def nthPrime (n : ℕ) : ℕ := sorry

def cumulativePrimeSum (n : ℕ) : ℕ := 
  if n = 0 then 0 else cumulativePrimeSum (n-1) + nthPrime (n+1)

theorem no_prime_in_first_15_cumulative_sums : 
  ∀ n : ℕ, n > 0 → n ≤ 15 → ¬(isPrime (cumulativePrimeSum n)) :=
sorry

end NUMINAMATH_CALUDE_no_prime_in_first_15_cumulative_sums_l1145_114507


namespace NUMINAMATH_CALUDE_max_3m_plus_4n_l1145_114564

theorem max_3m_plus_4n (m n : ℕ) : 
  (∃ (evens : Finset ℕ) (odds : Finset ℕ), 
    evens.card = m ∧ 
    odds.card = n ∧
    (∀ x ∈ evens, Even x ∧ x > 0) ∧
    (∀ y ∈ odds, Odd y ∧ y > 0) ∧
    (evens.sum id + odds.sum id = 1987)) →
  3 * m + 4 * n ≤ 221 :=
by sorry

end NUMINAMATH_CALUDE_max_3m_plus_4n_l1145_114564


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1145_114596

theorem fixed_point_on_line (m : ℝ) : 
  (3*m + 4) * (-1) + (5 - 2*m) * 2 + 7*m - 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1145_114596


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt3_l1145_114579

theorem sin_40_tan_10_minus_sqrt3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt3_l1145_114579


namespace NUMINAMATH_CALUDE_opposite_of_half_l1145_114562

-- Define the concept of opposite
def opposite (x : ℝ) : ℝ := -x

-- Theorem statement
theorem opposite_of_half : opposite 0.5 = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_half_l1145_114562


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1145_114540

theorem arithmetic_calculation : (-1 + 2) * 3 + 2^2 / (-4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1145_114540
