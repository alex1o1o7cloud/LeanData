import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l1886_188662

theorem sqrt_equation_solutions (x : ℝ) :
  Real.sqrt ((2 + Real.sqrt 5) ^ x) + Real.sqrt ((2 - Real.sqrt 5) ^ x) = 6 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l1886_188662


namespace NUMINAMATH_CALUDE_target_score_proof_l1886_188638

theorem target_score_proof (a b c : ℕ) 
  (h1 : 2 * b + c = 29) 
  (h2 : 2 * a + c = 43) : 
  a + b + c = 36 := by
sorry

end NUMINAMATH_CALUDE_target_score_proof_l1886_188638


namespace NUMINAMATH_CALUDE_negation_of_existence_sqrt_leq_x_minus_one_negation_l1886_188689

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem sqrt_leq_x_minus_one_negation :
  (¬ ∃ x > 0, Real.sqrt x ≤ x - 1) ↔ (∀ x > 0, Real.sqrt x > x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_sqrt_leq_x_minus_one_negation_l1886_188689


namespace NUMINAMATH_CALUDE_train_speed_time_reduction_l1886_188618

theorem train_speed_time_reduction :
  ∀ (v S : ℝ),
  v > 0 → S > 0 →
  let original_time := S / v
  let new_speed := 1.25 * v
  let new_time := S / new_speed
  (original_time - new_time) / original_time = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_time_reduction_l1886_188618


namespace NUMINAMATH_CALUDE_emily_sleep_duration_l1886_188617

/-- Calculates the time Emily slept during her flight -/
def time_emily_slept (flight_duration : ℕ) (num_episodes : ℕ) (episode_duration : ℕ) 
  (num_movies : ℕ) (movie_duration : ℕ) (remaining_time : ℕ) : ℚ :=
  let total_flight_minutes := flight_duration * 60
  let total_tv_minutes := num_episodes * episode_duration
  let total_movie_minutes := num_movies * movie_duration
  let sleep_minutes := total_flight_minutes - total_tv_minutes - total_movie_minutes - remaining_time
  (sleep_minutes : ℚ) / 60

/-- Theorem stating that Emily slept for 4.5 hours -/
theorem emily_sleep_duration :
  time_emily_slept 10 3 25 2 105 45 = 4.5 := by sorry

end NUMINAMATH_CALUDE_emily_sleep_duration_l1886_188617


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1886_188605

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1886_188605


namespace NUMINAMATH_CALUDE_radiator_antifreeze_percentage_l1886_188625

/-- The capacity of the radiator in liters -/
def radiator_capacity : ℝ := 6

/-- The volume of liquid replaced with pure antifreeze in liters -/
def replaced_volume : ℝ := 1

/-- The final percentage of antifreeze in the mixture -/
def final_percentage : ℝ := 0.5

/-- The initial percentage of antifreeze in the radiator -/
def initial_percentage : ℝ := 0.4

theorem radiator_antifreeze_percentage :
  let remaining_volume := radiator_capacity - replaced_volume
  let initial_antifreeze := initial_percentage * radiator_capacity
  let remaining_antifreeze := initial_antifreeze - initial_percentage * replaced_volume
  let final_antifreeze := remaining_antifreeze + replaced_volume
  final_antifreeze = final_percentage * radiator_capacity :=
by sorry

end NUMINAMATH_CALUDE_radiator_antifreeze_percentage_l1886_188625


namespace NUMINAMATH_CALUDE_divisibility_property_l1886_188670

theorem divisibility_property (a n p : ℕ) : 
  a ≥ 2 → 
  n ≥ 1 → 
  Nat.Prime p → 
  p ∣ (a^(2^n) + 1) → 
  2^(n+1) ∣ (p-1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1886_188670


namespace NUMINAMATH_CALUDE_divide_algebraic_expression_l1886_188628

theorem divide_algebraic_expression (a b : ℝ) (h : a ≠ 0) :
  (8 * a * b) / (2 * a) = 4 * b := by
  sorry

end NUMINAMATH_CALUDE_divide_algebraic_expression_l1886_188628


namespace NUMINAMATH_CALUDE_difference_empty_implies_subset_l1886_188601

-- Define the difference of two sets
def set_difference (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

-- State the theorem
theorem difference_empty_implies_subset {A B : Set α} :
  set_difference A B = ∅ → A ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_difference_empty_implies_subset_l1886_188601


namespace NUMINAMATH_CALUDE_dividing_line_slope_absolute_value_is_one_l1886_188660

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line that equally divides the total area of the circles --/
structure DividingLine where
  slope : ℝ
  passes_through : ℝ × ℝ

/-- The problem setup --/
def problem_setup : (Circle × Circle × Circle) × DividingLine := 
  let c1 : Circle := ⟨(10, 90), 4⟩
  let c2 : Circle := ⟨(15, 70), 4⟩
  let c3 : Circle := ⟨(20, 80), 4⟩
  let line : DividingLine := ⟨0, (15, 70)⟩  -- slope initialized to 0
  ((c1, c2, c3), line)

/-- The theorem to be proved --/
theorem dividing_line_slope_absolute_value_is_one 
  (setup : (Circle × Circle × Circle) × DividingLine) : 
  abs setup.2.slope = 1 := by
  sorry

end NUMINAMATH_CALUDE_dividing_line_slope_absolute_value_is_one_l1886_188660


namespace NUMINAMATH_CALUDE_given_segments_proportionate_l1886_188664

/-- A set of line segments is proportionate if the product of any two segments
    equals the product of the remaining two segments. -/
def IsProportionate (a b c d : ℝ) : Prop :=
  a * d = b * c

/-- The given set of line segments. -/
def LineSegments : (ℝ × ℝ × ℝ × ℝ) :=
  (3, 6, 4, 8)

/-- Theorem stating that the given set of line segments is proportionate. -/
theorem given_segments_proportionate :
  let (a, b, c, d) := LineSegments
  IsProportionate a b c d := by
  sorry

end NUMINAMATH_CALUDE_given_segments_proportionate_l1886_188664


namespace NUMINAMATH_CALUDE_root_ratio_theorem_l1886_188607

theorem root_ratio_theorem (k : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 + k*x₁ + 3/4*k^2 - 3*k + 9/2 = 0 →
  x₂^2 + k*x₂ + 3/4*k^2 - 3*k + 9/2 = 0 →
  x₁ ≠ x₂ →
  x₁^2020 / x₂^2021 = -2/3 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_theorem_l1886_188607


namespace NUMINAMATH_CALUDE_no_roots_of_third_polynomial_l1886_188602

/-- Given two quadratic polynomials with integer coefficients that have at least one integer root each,
    prove that a third polynomial with a constant term increased by 2 has no real roots. -/
theorem no_roots_of_third_polynomial (a b : ℤ) :
  (∃ x : ℤ, x^2 + a*x + b = 0) →
  (∃ y : ℤ, y^2 + a*y + (b + 1) = 0) →
  ∀ z : ℝ, z^2 + a*z + (b + 2) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_roots_of_third_polynomial_l1886_188602


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1886_188696

/-- Represents the number of ways to arrange books of two subjects. -/
def arrange_books (total : ℕ) (subject1 : ℕ) (subject2 : ℕ) : ℕ :=
  2 * 2 * 2

/-- Theorem stating that arranging 4 books (2 Chinese and 2 math) 
    such that books of the same subject are not adjacent 
    results in 8 possible arrangements. -/
theorem book_arrangement_theorem :
  arrange_books 4 2 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1886_188696


namespace NUMINAMATH_CALUDE_a_range_theorem_l1886_188606

-- Define the type for real numbers greater than zero
def PositiveReal := {x : ℝ // x > 0}

-- Define the monotonically increasing property for a^x
def MonotonicallyIncreasing (a : PositiveReal) : Prop :=
  ∀ x y : ℝ, x < y → (a.val : ℝ) ^ x < (a.val : ℝ) ^ y

-- Define the property that x^2 - ax + 1 > 0 does not hold for all x
def NotAlwaysPositive (a : PositiveReal) : Prop :=
  ¬(∀ x : ℝ, x^2 - (a.val : ℝ) * x + 1 > 0)

-- State the theorem
theorem a_range_theorem (a : PositiveReal) 
  (h1 : MonotonicallyIncreasing a) 
  (h2 : NotAlwaysPositive a) : 
  (a.val : ℝ) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_a_range_theorem_l1886_188606


namespace NUMINAMATH_CALUDE_range_of_a_l1886_188674

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ∈ Set.union (Set.Iic (-2)) {1} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1886_188674


namespace NUMINAMATH_CALUDE_radius_of_touching_sphere_l1886_188690

/-- A regular quadrilateral pyramid with an inscribed sphere and a touching sphere -/
structure PyramidWithSpheres where
  -- Base side length
  a : ℝ
  -- Lateral edge length
  b : ℝ
  -- Radius of inscribed sphere Q₁
  r₁ : ℝ
  -- Radius of touching sphere Q₂
  r₂ : ℝ
  -- Condition: The pyramid is regular quadrilateral
  regular : a > 0 ∧ b > 0
  -- Condition: Q₁ is inscribed in the pyramid
  q₁_inscribed : r₁ > 0
  -- Condition: Q₂ touches Q₁ and all lateral faces
  q₂_touches : r₂ > 0

/-- Theorem stating the radius of Q₂ in the given pyramid configuration -/
theorem radius_of_touching_sphere (p : PyramidWithSpheres) 
  (h₁ : p.a = 6) 
  (h₂ : p.b = 5) : 
  p.r₂ = 3 * Real.sqrt 7 / 49 := by
  sorry


end NUMINAMATH_CALUDE_radius_of_touching_sphere_l1886_188690


namespace NUMINAMATH_CALUDE_circle_condition_l1886_188661

/-- The equation of a potential circle -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + x + 2*m*y + m^2 + m - 1 = 0

/-- Theorem stating the condition for the equation to represent a circle -/
theorem circle_condition (m : ℝ) :
  (∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2) ↔
  m < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l1886_188661


namespace NUMINAMATH_CALUDE_equation_solution_l1886_188687

theorem equation_solution : ∃ x : ℝ, 3^(x - 1) = (1 : ℝ) / 9 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1886_188687


namespace NUMINAMATH_CALUDE_doughnuts_served_l1886_188634

theorem doughnuts_served (staff : ℕ) (doughnuts_per_staff : ℕ) (doughnuts_left : ℕ) : 
  staff = 19 → doughnuts_per_staff = 2 → doughnuts_left = 12 →
  staff * doughnuts_per_staff + doughnuts_left = 50 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_served_l1886_188634


namespace NUMINAMATH_CALUDE_cubic_roots_inequality_l1886_188697

theorem cubic_roots_inequality (a b c : ℝ) : 
  (∃ x y z : ℝ, ∀ t : ℝ, t^3 + a*t^2 + b*t + c = (t - x) * (t - y) * (t - z)) → 
  3*b ≤ a^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_inequality_l1886_188697


namespace NUMINAMATH_CALUDE_performance_arrangement_count_l1886_188675

/-- The number of ways to arrange n elements from a set of k elements --/
def A (k n : ℕ) : ℕ := sorry

/-- The number of ways to choose n elements from a set of k elements, where order matters --/
def P (k n : ℕ) : ℕ := sorry

/-- The number of ways to arrange 6 singing programs and 4 dance programs, 
    where no two dance programs can be adjacent --/
def arrangement_count : ℕ := P 7 4 * A 6 6

theorem performance_arrangement_count : 
  arrangement_count = P 7 4 * A 6 6 := by sorry

end NUMINAMATH_CALUDE_performance_arrangement_count_l1886_188675


namespace NUMINAMATH_CALUDE_number_of_cat_only_owners_cat_only_owners_count_l1886_188616

theorem number_of_cat_only_owners (total_pet_owners : ℕ) (only_dog_owners : ℕ) 
  (cat_and_dog_owners : ℕ) (cat_dog_snake_owners : ℕ) (total_snakes : ℕ) : ℕ :=
  let snake_only_owners := total_snakes - cat_dog_snake_owners
  let cat_only_owners := total_pet_owners - only_dog_owners - cat_and_dog_owners - 
                         cat_dog_snake_owners - snake_only_owners
  cat_only_owners

theorem cat_only_owners_count : 
  number_of_cat_only_owners 69 15 5 3 39 = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cat_only_owners_cat_only_owners_count_l1886_188616


namespace NUMINAMATH_CALUDE_elastic_collision_mass_and_velocity_ratios_l1886_188611

/-- Represents the masses and velocities in an elastic collision -/
structure CollisionSystem where
  m₁ : ℝ
  m₂ : ℝ
  v₀ : ℝ
  v₁ : ℝ
  v₂ : ℝ

/-- Conditions for the elastic collision system -/
def ElasticCollision (s : CollisionSystem) : Prop :=
  s.m₁ > 0 ∧ s.m₂ > 0 ∧ s.v₀ > 0 ∧ s.v₁ > 0 ∧ s.v₂ > 0 ∧
  s.v₂ = 4 * s.v₁ ∧
  s.m₁ * s.v₀ = s.m₁ * s.v₁ + s.m₂ * s.v₂ ∧
  s.m₁ * s.v₀^2 = s.m₁ * s.v₁^2 + s.m₂ * s.v₂^2

theorem elastic_collision_mass_and_velocity_ratios (s : CollisionSystem) 
  (h : ElasticCollision s) : s.m₂ / s.m₁ = 1/2 ∧ s.v₀ / s.v₁ = 3 := by
  sorry

end NUMINAMATH_CALUDE_elastic_collision_mass_and_velocity_ratios_l1886_188611


namespace NUMINAMATH_CALUDE_sams_first_month_earnings_l1886_188640

/-- Sam's hourly rate for Math tutoring -/
def hourly_rate : ℕ := 10

/-- The difference in earnings between the second and first month -/
def second_month_increase : ℕ := 150

/-- Total hours spent tutoring over two months -/
def total_hours : ℕ := 55

/-- Sam's earnings in the first month -/
def first_month_earnings : ℕ := 200

/-- Theorem stating that Sam's earnings in the first month were $200 -/
theorem sams_first_month_earnings :
  first_month_earnings = (hourly_rate * total_hours - second_month_increase) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sams_first_month_earnings_l1886_188640


namespace NUMINAMATH_CALUDE_max_product_sum_320_l1886_188698

theorem max_product_sum_320 : 
  ∃ (a b : ℤ), a + b = 320 ∧ 
  ∀ (x y : ℤ), x + y = 320 → x * y ≤ a * b ∧
  a * b = 25600 := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_320_l1886_188698


namespace NUMINAMATH_CALUDE_negation_equivalence_l1886_188666

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≥ 1 ∨ x > 2) ↔ (∀ x : ℝ, x < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1886_188666


namespace NUMINAMATH_CALUDE_amy_book_count_l1886_188630

theorem amy_book_count (maddie_books luisa_books : ℕ) 
  (h1 : maddie_books = 15)
  (h2 : luisa_books = 18)
  (h3 : luisa_books + amy_books = maddie_books + 9) : 
  amy_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_amy_book_count_l1886_188630


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1886_188695

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_roots : a 4 > 0 ∧ a 8 > 0 ∧ a 4^2 - 4*a 4 + 3 = 0 ∧ a 8^2 - 4*a 8 + 3 = 0) :
  a 6 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1886_188695


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l1886_188649

/-- 
Given that Jose starts with some bottle caps, gets 2 more from Rebecca, 
and ends up with 9 bottle caps, prove that he started with 7 bottle caps.
-/
theorem jose_bottle_caps (x : ℕ) : x + 2 = 9 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l1886_188649


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_is_79_l1886_188633

/-- The expression to be simplified -/
def expression (y : ℝ) : ℝ := 5 * (y^2 - 3*y + 3) - 6 * (y^3 - 2*y + 2)

/-- The sum of squares of coefficients of the simplified expression -/
def sum_of_squares_of_coefficients : ℕ := 79

/-- Theorem stating that the sum of squares of coefficients of the simplified expression is 79 -/
theorem sum_of_squares_of_coefficients_is_79 : 
  sum_of_squares_of_coefficients = 79 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_is_79_l1886_188633


namespace NUMINAMATH_CALUDE_georges_initial_socks_l1886_188610

theorem georges_initial_socks (bought new_from_dad total_now : ℕ) 
  (h1 : bought = 36)
  (h2 : new_from_dad = 4)
  (h3 : total_now = 68)
  : total_now - bought - new_from_dad = 28 := by
  sorry

end NUMINAMATH_CALUDE_georges_initial_socks_l1886_188610


namespace NUMINAMATH_CALUDE_ramsey_r33_l1886_188645

-- Define a type for the colors of the edges
inductive Color
| Red
| Blue

-- Define the graph type
def Graph := Fin 6 → Fin 6 → Color

-- Define what it means for three vertices to form a monochromatic triangle
def IsMonochromaticTriangle (g : Graph) (v1 v2 v3 : Fin 6) : Prop :=
  g v1 v2 = g v2 v3 ∧ g v2 v3 = g v3 v1

-- State the theorem
theorem ramsey_r33 (g : Graph) :
  (∀ (v1 v2 : Fin 6), v1 ≠ v2 → g v1 v2 = g v2 v1) →  -- Symmetry condition
  (∃ (v1 v2 v3 : Fin 6), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ IsMonochromaticTriangle g v1 v2 v3) :=
by
  sorry

end NUMINAMATH_CALUDE_ramsey_r33_l1886_188645


namespace NUMINAMATH_CALUDE_prob_different_colors_bag_l1886_188683

/-- Represents the number of chips of each color in the bag -/
structure ChipCounts where
  blue : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of drawing two chips of different colors -/
def probDifferentColors (counts : ChipCounts) : ℚ :=
  let total := counts.blue + counts.red + counts.yellow + counts.green
  let probBlue := counts.blue / total
  let probRed := counts.red / total
  let probYellow := counts.yellow / total
  let probGreen := counts.green / total
  let probDiffAfterBlue := (total - counts.blue) / total
  let probDiffAfterRed := (total - counts.red) / total
  let probDiffAfterYellow := (total - counts.yellow) / total
  let probDiffAfterGreen := (total - counts.green) / total
  probBlue * probDiffAfterBlue + probRed * probDiffAfterRed +
  probYellow * probDiffAfterYellow + probGreen * probDiffAfterGreen

/-- The main theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_bag :
  probDifferentColors { blue := 6, red := 5, yellow := 4, green := 3 } = 119 / 162 := by
  sorry


end NUMINAMATH_CALUDE_prob_different_colors_bag_l1886_188683


namespace NUMINAMATH_CALUDE_lilly_and_rosy_fish_l1886_188648

/-- The number of fish Lilly and Rosy have together -/
def total_fish (lilly_fish rosy_fish : ℕ) : ℕ := lilly_fish + rosy_fish

/-- Theorem: Lilly and Rosy have 22 fish in total -/
theorem lilly_and_rosy_fish : total_fish 10 12 = 22 := by
  sorry

end NUMINAMATH_CALUDE_lilly_and_rosy_fish_l1886_188648


namespace NUMINAMATH_CALUDE_discounted_shoe_price_l1886_188655

/-- Given a pair of shoes bought at a 20% discount for $480, 
    prove that the original price was $600. -/
theorem discounted_shoe_price (discount_rate : ℝ) (discounted_price : ℝ) :
  discount_rate = 0.20 →
  discounted_price = 480 →
  discounted_price = (1 - discount_rate) * 600 :=
by sorry

end NUMINAMATH_CALUDE_discounted_shoe_price_l1886_188655


namespace NUMINAMATH_CALUDE_petes_flag_total_shapes_l1886_188612

/-- The number of stars on the US flag -/
def us_stars : ℕ := 50

/-- The number of stripes on the US flag -/
def us_stripes : ℕ := 13

/-- The number of circles on Pete's flag -/
def petes_circles : ℕ := us_stars / 2 - 3

/-- The number of squares on Pete's flag -/
def petes_squares : ℕ := us_stripes * 2 + 6

/-- Theorem stating the total number of shapes on Pete's flag -/
theorem petes_flag_total_shapes :
  petes_circles + petes_squares = 54 := by sorry

end NUMINAMATH_CALUDE_petes_flag_total_shapes_l1886_188612


namespace NUMINAMATH_CALUDE_expression_not_33_l1886_188677

theorem expression_not_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_not_33_l1886_188677


namespace NUMINAMATH_CALUDE_polynomial_M_proof_l1886_188679

-- Define the polynomial M as a function of x and y
def M (x y : ℝ) : ℝ := 2 * x * y - 1

-- Theorem statement
theorem polynomial_M_proof :
  -- Given condition
  (∀ x y : ℝ, M x y + (2 * x^2 * y - 3 * x * y + 1) = 2 * x^2 * y - x * y) →
  -- Conclusion 1: M is correctly defined
  (∀ x y : ℝ, M x y = 2 * x * y - 1) ∧
  -- Conclusion 2: M(-1, 2) = -5
  (M (-1) 2 = -5) := by
sorry

end NUMINAMATH_CALUDE_polynomial_M_proof_l1886_188679


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1886_188626

/-- The decimal representation of 0.456̄ as a rational number -/
def repeating_decimal : ℚ := 456 / 999

theorem product_of_repeating_decimal_and_eight :
  repeating_decimal * 8 = 1216 / 333 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1886_188626


namespace NUMINAMATH_CALUDE_grandmother_mother_age_ratio_l1886_188646

/-- Represents the ages of Grace, her mother, and her grandmother -/
structure FamilyAges where
  grace : ℕ
  mother : ℕ
  grandmother : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.grace = 60 ∧
  ages.mother = 80 ∧
  ages.grace * 8 = ages.grandmother * 3 ∧
  ∃ k : ℕ, ages.grandmother = k * ages.mother

/-- The theorem to be proved -/
theorem grandmother_mother_age_ratio 
  (ages : FamilyAges) 
  (h : problem_conditions ages) : 
  ages.grandmother / ages.mother = 2 := by
  sorry


end NUMINAMATH_CALUDE_grandmother_mother_age_ratio_l1886_188646


namespace NUMINAMATH_CALUDE_convex_polygon_20_sides_diagonals_l1886_188663

/-- A convex polygon is a polygon in which every interior angle is less than 180 degrees. -/
def ConvexPolygon (n : ℕ) : Prop := sorry

/-- A diagonal of a convex polygon is a line segment that connects two non-adjacent vertices. -/
def Diagonal (n : ℕ) : Prop := sorry

/-- The number of diagonals in a convex polygon with n sides. -/
def NumDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem convex_polygon_20_sides_diagonals :
  ∀ p : ConvexPolygon 20, NumDiagonals 20 = 170 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_20_sides_diagonals_l1886_188663


namespace NUMINAMATH_CALUDE_largest_non_60multiple_composite_sum_l1886_188639

/-- A positive integer is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A function that represents the sum of a positive integral multiple of 60 and a positive composite integer -/
def SumOf60MultipleAndComposite (k m : ℕ) : ℕ := 60 * (k + 1) + m

theorem largest_non_60multiple_composite_sum :
  ∀ n : ℕ, n > 5 →
    ∃ k m : ℕ, IsComposite m ∧ n = SumOf60MultipleAndComposite k m :=
by sorry

end NUMINAMATH_CALUDE_largest_non_60multiple_composite_sum_l1886_188639


namespace NUMINAMATH_CALUDE_remainder_after_adding_1008_l1886_188682

theorem remainder_after_adding_1008 (n : ℤ) : 
  n % 4 = 1 → (n + 1008) % 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_after_adding_1008_l1886_188682


namespace NUMINAMATH_CALUDE_natural_number_puzzle_l1886_188642

def first_digit (n : ℕ) : ℕ := n.div (10 ^ (n.log 10))

def last_digit (n : ℕ) : ℕ := n % 10

def swap_first_last (n : ℕ) : ℕ :=
  let d := n.log 10
  last_digit n * 10^d + (n - first_digit n * 10^d - last_digit n) + first_digit n

theorem natural_number_puzzle (x : ℕ) :
  first_digit x = 2 →
  last_digit x = 5 →
  swap_first_last x = 2 * x + 2 →
  x ≤ 10000 →
  x = 25 ∨ x = 295 ∨ x = 2995 := by
  sorry

end NUMINAMATH_CALUDE_natural_number_puzzle_l1886_188642


namespace NUMINAMATH_CALUDE_no_permutations_satisfy_condition_l1886_188619

theorem no_permutations_satisfy_condition :
  ∀ (b₁ b₂ b₃ b₄ b₅ b₆ : ℕ), 
    b₁ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₂ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₃ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₄ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₅ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₆ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
    b₁ ≠ b₂ ∧ b₁ ≠ b₃ ∧ b₁ ≠ b₄ ∧ b₁ ≠ b₅ ∧ b₁ ≠ b₆ ∧
    b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧
    b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧
    b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧
    b₅ ≠ b₆ →
    ((b₁ + 1) / 3) * ((b₂ + 2) / 3) * ((b₃ + 3) / 3) * 
    ((b₄ + 4) / 3) * ((b₅ + 5) / 3) * ((b₆ + 6) / 3) ≤ 120 := by
  sorry


end NUMINAMATH_CALUDE_no_permutations_satisfy_condition_l1886_188619


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1886_188654

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : a 1 = 1
  h3 : ∀ n : ℕ, a (n + 1) = a n + d
  h4 : (a 5) ^ 2 = (a 3) * (a 10)

/-- The sum of the first n terms of the arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1) + (n * (n - 1) * seq.d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (n : ℕ) :
  sum_n seq n = -3/4 * n^2 + 7/4 * n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1886_188654


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1886_188644

/-- Represents a hyperbola with a given asymptote and a point it passes through -/
structure Hyperbola where
  asymptote_slope : ℝ
  point : ℝ × ℝ

/-- The standard form of a hyperbola equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the standard equation of a hyperbola given its asymptote and a point -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 1/2)
    (h_point : h.point = (2 * Real.sqrt 2, 1)) :
    standard_equation 4 1 (h.point.1) (h.point.2) :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1886_188644


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l1886_188672

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem reflection_across_x_axis :
  let M : Point3D := { x := -1, y := 2, z := 1 }
  reflect_x_axis M = { x := -1, y := -2, z := -1 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l1886_188672


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l1886_188615

theorem missing_digit_divisible_by_three (x : Nat) :
  (x < 10) →
  (246 * 100 + x * 10 + 9) % 3 = 0 →
  x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l1886_188615


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_l1886_188685

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the perimeter of a square -/
def Square.perimeter (s : Square) : ℝ := 4 * s.side

/-- Theorem stating the perimeter of the large rectangle -/
theorem large_rectangle_perimeter 
  (square : Square) 
  (small_rect : Rectangle) 
  (h1 : square.perimeter = 24) 
  (h2 : small_rect.perimeter = 16) 
  (h3 : small_rect.length = square.side) :
  let large_rect := Rectangle.mk (3 * square.side + small_rect.length) (square.side + small_rect.width)
  large_rect.perimeter = 52 := by
  sorry


end NUMINAMATH_CALUDE_large_rectangle_perimeter_l1886_188685


namespace NUMINAMATH_CALUDE_mary_shirts_left_l1886_188657

/-- The number of shirts Mary has left after giving away some -/
def shirts_left (blue_shirts : ℕ) (brown_shirts : ℕ) : ℕ :=
  blue_shirts - blue_shirts / 2 + brown_shirts - brown_shirts / 3

/-- Theorem: Mary has 37 shirts left -/
theorem mary_shirts_left : shirts_left 26 36 = 37 := by
  sorry

end NUMINAMATH_CALUDE_mary_shirts_left_l1886_188657


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1886_188641

theorem unique_three_digit_number : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ 
  (300 + n / 10) = 3 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1886_188641


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1886_188622

/-- Given two parallel vectors a and b in R², prove that the magnitude of b is 2√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, 2) → 
  b.1 = -2 → 
  (a.1 * b.2 = a.2 * b.1) → 
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1886_188622


namespace NUMINAMATH_CALUDE_square_ratios_l1886_188629

theorem square_ratios (a b : ℝ) (h : b = 3 * a) :
  (4 * b) / (4 * a) = 3 ∧ (b * b) / (a * a) = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_ratios_l1886_188629


namespace NUMINAMATH_CALUDE_tank_capacity_l1886_188658

/-- Represents the tank system with leaks and pipes -/
structure TankSystem where
  capacity : ℝ
  leak1Rate : ℝ
  leak2Rate : ℝ
  inletRate : ℝ
  emptyTime : ℝ

/-- The tank system satisfies the given conditions -/
def validTankSystem (t : TankSystem) : Prop :=
  t.leak1Rate = t.capacity / 4 ∧
  t.leak2Rate = t.capacity / 8 ∧
  t.inletRate = 360 ∧
  t.emptyTime = 12 ∧
  t.inletRate - t.leak1Rate - t.leak2Rate = t.capacity / t.emptyTime

/-- The theorem stating that a valid tank system has a capacity of 785 litres -/
theorem tank_capacity (t : TankSystem) (h : validTankSystem t) : 
  t.capacity = 785 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l1886_188658


namespace NUMINAMATH_CALUDE_balloon_ratio_l1886_188673

theorem balloon_ratio : 
  let gold_balloons : ℕ := 141
  let black_balloons : ℕ := 150
  let total_balloons : ℕ := 573
  let silver_balloons : ℕ := total_balloons - gold_balloons - black_balloons
  (silver_balloons : ℚ) / gold_balloons = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_ratio_l1886_188673


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1886_188624

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → 
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1886_188624


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1886_188637

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| < 3} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1886_188637


namespace NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_to_1001_l1886_188671

theorem sum_of_tens_and_units_digits_of_9_to_1001 :
  ∃ (n : ℕ), 9^1001 = 100 * n + 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_tens_and_units_digits_of_9_to_1001_l1886_188671


namespace NUMINAMATH_CALUDE_square_of_rational_difference_l1886_188667

theorem square_of_rational_difference (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by sorry

end NUMINAMATH_CALUDE_square_of_rational_difference_l1886_188667


namespace NUMINAMATH_CALUDE_four_weavers_four_days_l1886_188650

/-- The number of mats woven by a group of weavers over a period of days. -/
def mats_woven (weavers : ℕ) (days : ℕ) : ℚ :=
  (25 : ℚ) * weavers * days / (10 * 10)

/-- Theorem stating that 4 mat-weavers will weave 4 mats in 4 days given the rate
    at which 10 mat-weavers can weave 25 mats in 10 days. -/
theorem four_weavers_four_days :
  mats_woven 4 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_weavers_four_days_l1886_188650


namespace NUMINAMATH_CALUDE_sequence_term_correct_l1886_188631

def sequence_sum (n : ℕ) : ℕ := 3 + 2^n

def sequence_term (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2^(n-1)

theorem sequence_term_correct :
  ∀ n : ℕ, n ≥ 1 →
  sequence_sum n - sequence_sum (n-1) = sequence_term n :=
sorry

end NUMINAMATH_CALUDE_sequence_term_correct_l1886_188631


namespace NUMINAMATH_CALUDE_initial_violet_balloons_count_l1886_188653

/-- The number of violet balloons Jason initially had -/
def initial_violet_balloons : ℕ := sorry

/-- The number of violet balloons Jason lost -/
def lost_violet_balloons : ℕ := 3

/-- The number of violet balloons Jason has now -/
def current_violet_balloons : ℕ := 4

/-- Theorem stating that the initial number of violet balloons is 7 -/
theorem initial_violet_balloons_count : initial_violet_balloons = 7 :=
by
  sorry

/-- Lemma showing the relationship between initial, lost, and current balloons -/
lemma balloon_relationship : initial_violet_balloons = current_violet_balloons + lost_violet_balloons :=
by
  sorry

end NUMINAMATH_CALUDE_initial_violet_balloons_count_l1886_188653


namespace NUMINAMATH_CALUDE_competitive_exam_selection_difference_l1886_188600

theorem competitive_exam_selection_difference (total_candidates : ℕ) 
  (selection_rate_A : ℚ) (selection_rate_B : ℚ) : 
  total_candidates = 7900 → 
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B - selection_rate_A) * total_candidates = 79 := by
sorry

end NUMINAMATH_CALUDE_competitive_exam_selection_difference_l1886_188600


namespace NUMINAMATH_CALUDE_summer_sun_salutations_l1886_188609

/-- The number of sun salutations Summer performs in a year -/
def sun_salutations_per_year (poses_per_day : ℕ) (weekdays_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  poses_per_day * weekdays_per_week * weeks_per_year

/-- Theorem: Summer performs 1300 sun salutations in a year -/
theorem summer_sun_salutations :
  sun_salutations_per_year 5 5 52 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_summer_sun_salutations_l1886_188609


namespace NUMINAMATH_CALUDE_expression_simplification_l1886_188621

theorem expression_simplification (x y : ℚ) 
  (hx : x = -1/2) (hy : y = -3) : 
  3 * (x^2 - 2*x*y) - (3*x^2 - 2*y + 2*(x*y + y)) = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1886_188621


namespace NUMINAMATH_CALUDE_only_valid_pythagorean_triple_l1886_188659

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem only_valid_pythagorean_triple :
  ¬ is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 2 2 (2 * 2) ∧
  ¬ is_pythagorean_triple 4 5 6 ∧
  is_pythagorean_triple 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_only_valid_pythagorean_triple_l1886_188659


namespace NUMINAMATH_CALUDE_train_speed_train_speed_is_50_l1886_188665

/-- The speed of a train given travel time and alternative speed scenario -/
theorem train_speed (travel_time : ℝ) (alt_time : ℝ) (alt_speed : ℝ) : ℝ :=
  let distance := alt_speed * alt_time
  distance / travel_time

/-- Proof that the train's speed is 50 mph given the specified conditions -/
theorem train_speed_is_50 :
  train_speed 4 2 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_is_50_l1886_188665


namespace NUMINAMATH_CALUDE_taqeeshas_grade_l1886_188668

theorem taqeeshas_grade (total_students : ℕ) (students_present : ℕ) (initial_average : ℕ) (final_average : ℕ) :
  total_students = 17 →
  students_present = 16 →
  initial_average = 77 →
  final_average = 78 →
  (students_present * initial_average + (total_students - students_present) * 94) / total_students = final_average :=
by sorry

end NUMINAMATH_CALUDE_taqeeshas_grade_l1886_188668


namespace NUMINAMATH_CALUDE_quadrilaterals_form_polygons_l1886_188614

/-- A point in 2D space --/
structure Point :=
  (x : ℤ)
  (y : ℤ)

/-- A polygon defined by its vertices --/
structure Polygon :=
  (vertices : List Point)

/-- Definition of a square --/
def is_square (p : Polygon) : Prop :=
  p.vertices.length = 4 ∧
  ∃ (x y : ℤ), p.vertices = [Point.mk x y, Point.mk (x+2) y, Point.mk (x+2) (y+2), Point.mk x (y+2)]

/-- Definition of a triangle --/
def is_triangle (p : Polygon) : Prop :=
  p.vertices.length = 3

/-- Definition of a pentagon --/
def is_pentagon (p : Polygon) : Prop :=
  p.vertices.length = 5

/-- The two squares from the problem --/
def square1 : Polygon :=
  Polygon.mk [Point.mk 0 0, Point.mk 2 0, Point.mk 2 2, Point.mk 0 2]

def square2 : Polygon :=
  Polygon.mk [Point.mk 2 2, Point.mk 4 2, Point.mk 4 4, Point.mk 2 4]

/-- Main theorem --/
theorem quadrilaterals_form_polygons :
  (is_square square1 ∧ is_square square2) →
  (∃ (t : Polygon) (p : Polygon), is_triangle t ∧ is_pentagon p ∧
    (∀ v : Point, v ∈ t.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices) ∧
    (∀ v : Point, v ∈ p.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices)) ∧
  (∃ (t : Polygon) (q : Polygon) (p : Polygon), 
    is_triangle t ∧ p.vertices.length = 4 ∧ is_pentagon p ∧
    (∀ v : Point, v ∈ t.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices) ∧
    (∀ v : Point, v ∈ q.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices) ∧
    (∀ v : Point, v ∈ p.vertices → v ∈ square1.vertices ∨ v ∈ square2.vertices)) :=
by sorry

end NUMINAMATH_CALUDE_quadrilaterals_form_polygons_l1886_188614


namespace NUMINAMATH_CALUDE_range_of_a_for_nonempty_solution_set_l1886_188623

theorem range_of_a_for_nonempty_solution_set :
  (∃ (a : ℝ), ∃ (x : ℝ), |x + 2| + |x| ≤ a) →
  (∀ (a : ℝ), (∃ (x : ℝ), |x + 2| + |x| ≤ a) ↔ a ∈ Set.Ici 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_nonempty_solution_set_l1886_188623


namespace NUMINAMATH_CALUDE_john_weekly_loss_l1886_188678

/-- Represents John's tire production and sales scenario -/
structure TireProduction where
  daily_production : ℕ
  production_cost : ℚ
  selling_price_multiplier : ℚ
  potential_daily_sales : ℕ

/-- Calculates the weekly loss due to production limitations -/
def weekly_loss (t : TireProduction) : ℚ :=
  let profit_per_tire := t.production_cost * (t.selling_price_multiplier - 1)
  let daily_loss := profit_per_tire * (t.potential_daily_sales - t.daily_production)
  7 * daily_loss

/-- Theorem stating that given John's production scenario, the weekly loss is $175,000 -/
theorem john_weekly_loss :
  let john_production : TireProduction := {
    daily_production := 1000,
    production_cost := 250,
    selling_price_multiplier := 1.5,
    potential_daily_sales := 1200
  }
  weekly_loss john_production = 175000 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_loss_l1886_188678


namespace NUMINAMATH_CALUDE_tangent_point_on_parabola_l1886_188613

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + x - 2

-- Define the derivative of the parabola function
def f' (x : ℝ) : ℝ := 2*x + 1

-- Theorem statement
theorem tangent_point_on_parabola :
  let M : ℝ × ℝ := (1, 0)
  f M.1 = M.2 ∧ f' M.1 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_point_on_parabola_l1886_188613


namespace NUMINAMATH_CALUDE_algebraic_fraction_simplification_l1886_188647

theorem algebraic_fraction_simplification (x : ℝ) 
  (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 3*x + 2) / ((x^2 - 6*x + 9) / (x^2 - 7*x + 10)) = (x - 5) / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_fraction_simplification_l1886_188647


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l1886_188604

theorem cos_2alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/3) = 2/3) : 
  Real.cos (2*α + π/3) = -1/9 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l1886_188604


namespace NUMINAMATH_CALUDE_tan_inequality_l1886_188669

-- Define the constants and their properties
axiom α : Real
axiom β : Real
axiom k : Int

-- Define the conditions
axiom sin_inequality : Real.sin α > Real.sin β
axiom α_not_right_angle : ∀ k, α ≠ k * Real.pi + Real.pi / 2
axiom β_not_right_angle : ∀ k, β ≠ k * Real.pi + Real.pi / 2
axiom fourth_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi ∧ 3 * Real.pi / 2 < β ∧ β < 2 * Real.pi

-- State the theorem
theorem tan_inequality : Real.tan α > Real.tan β := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l1886_188669


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l1886_188686

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l1886_188686


namespace NUMINAMATH_CALUDE_sum_of_squares_roots_l1886_188632

theorem sum_of_squares_roots (h : ℝ) : 
  (∃ r s : ℝ, r^2 - 4*h*r - 8 = 0 ∧ s^2 - 4*h*s - 8 = 0 ∧ r^2 + s^2 = 20) → 
  h = 1/2 ∨ h = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_roots_l1886_188632


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l1886_188627

def fair_coin_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

theorem coin_flip_probability_difference :
  fair_coin_probability 4 3 - fair_coin_probability 4 4 = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l1886_188627


namespace NUMINAMATH_CALUDE_new_person_weight_l1886_188652

/-- Proves that if replacing a 50 kg person with a new person in a group of 5 
    increases the average weight by 4 kg, then the new person weighs 70 kg. -/
theorem new_person_weight (W : ℝ) : 
  W - 50 + (W + 20) / 5 = W + 4 → (W + 20) / 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1886_188652


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1886_188693

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The large box dimensions -/
def largeBox : BoxDimensions := ⟨6, 3, 4⟩

/-- The small block dimensions -/
def smallBlock : BoxDimensions := ⟨3, 1, 2⟩

/-- Calculates the volume of a box given its dimensions -/
def volume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Theorem: The maximum number of small blocks that can fit in the large box is 12 -/
theorem max_blocks_fit : 
  (volume largeBox) / (volume smallBlock) = 12 ∧ 
  largeBox.length / smallBlock.length * 
  largeBox.width / smallBlock.width * 
  largeBox.height / smallBlock.height = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1886_188693


namespace NUMINAMATH_CALUDE_train_length_l1886_188636

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 56 → time_s = 9 → ∃ (length_m : ℝ), 
  (abs (length_m - (speed_kmh * 1000 / 3600 * time_s)) < 0.01) ∧ 
  (abs (length_m - 140) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1886_188636


namespace NUMINAMATH_CALUDE_f_min_at_neg_two_l1886_188699

/-- The polynomial f(x) = x^2 + 4x + 4 -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The minimum value of f occurs at x = -2 -/
theorem f_min_at_neg_two :
  ∀ x : ℝ, f x ≥ f (-2) :=
by
  sorry

end NUMINAMATH_CALUDE_f_min_at_neg_two_l1886_188699


namespace NUMINAMATH_CALUDE_boat_fee_ratio_l1886_188656

/-- Proves that the ratio of docking fees to license and registration fees is 3:1 given the conditions of Mitch's boat purchase. -/
theorem boat_fee_ratio :
  let total_savings : ℚ := 20000
  let boat_cost_per_foot : ℚ := 1500
  let license_fee : ℚ := 500
  let max_boat_length : ℚ := 12
  let available_for_boat : ℚ := total_savings - license_fee
  let boat_cost : ℚ := boat_cost_per_foot * max_boat_length
  let docking_fee : ℚ := available_for_boat - boat_cost
  docking_fee / license_fee = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_fee_ratio_l1886_188656


namespace NUMINAMATH_CALUDE_star_example_l1886_188643

-- Define the star operation
def star (a b c d : ℚ) : ℚ := a * c * (d / b)

-- Theorem statement
theorem star_example : star (5/9) (4/6) = 40/3 := by sorry

end NUMINAMATH_CALUDE_star_example_l1886_188643


namespace NUMINAMATH_CALUDE_number_composition_l1886_188684

def number_from_parts (ten_millions hundreds_thousands hundreds : ℕ) : ℕ :=
  ten_millions * 10000000 + hundreds_thousands * 100000 + hundreds * 100

theorem number_composition :
  number_from_parts 4 6 5 = 46000500 := by sorry

end NUMINAMATH_CALUDE_number_composition_l1886_188684


namespace NUMINAMATH_CALUDE_gcd_equality_exists_l1886_188691

theorem gcd_equality_exists : ∃ k : ℕ+, 
  Nat.gcd 2012 2020 = Nat.gcd (2012 + k) 2020 ∧
  Nat.gcd 2012 2020 = Nat.gcd 2012 (2020 + k) ∧
  Nat.gcd 2012 2020 = Nat.gcd (2012 + k) (2020 + k) := by
  sorry

end NUMINAMATH_CALUDE_gcd_equality_exists_l1886_188691


namespace NUMINAMATH_CALUDE_power_inequality_l1886_188676

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^5 + b^5 > a^2 * b^3 + a^3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1886_188676


namespace NUMINAMATH_CALUDE_weight_difference_l1886_188680

/-- Given the weights of three people (Ishmael, Ponce, and Jalen), prove that Ishmael is 20 pounds heavier than Ponce. -/
theorem weight_difference (I P J : ℝ) : 
  J = 160 →  -- Jalen's weight
  P = J - 10 →  -- Ponce is 10 pounds lighter than Jalen
  (I + P + J) / 3 = 160 →  -- Average weight is 160 pounds
  I - P = 20 :=  -- Ishmael is 20 pounds heavier than Ponce
by sorry

end NUMINAMATH_CALUDE_weight_difference_l1886_188680


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1886_188651

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 4*a^2 + a + 1) * (b^3 + 4*b^2 + b + 1) * (c^3 + 4*c^2 + c + 1) / (a * b * c) ≥ 512 ∧
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    (x^3 + 4*x^2 + x + 1) * (y^3 + 4*y^2 + y + 1) * (z^3 + 4*z^2 + z + 1) / (x * y * z) = 512) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1886_188651


namespace NUMINAMATH_CALUDE_train_speed_l1886_188635

/-- Proves that the speed of a train is 36 km/hr given specific conditions -/
theorem train_speed (jogger_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  initial_distance = 240 →
  train_length = 120 →
  passing_time = 36 →
  (initial_distance + train_length) / passing_time * 3.6 = 36 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1886_188635


namespace NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l1886_188694

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a + 1, a - 1)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem range_of_a_in_fourth_quadrant :
  ∀ a : ℝ, in_fourth_quadrant (P a) ↔ -1 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l1886_188694


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l1886_188688

/-- Proves that the volume of fuel A added is 82 gallons given the specified conditions -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_A : ℝ) (ethanol_B : ℝ) (total_ethanol : ℝ) 
  (h1 : tank_capacity = 208)
  (h2 : ethanol_A = 0.12)
  (h3 : ethanol_B = 0.16)
  (h4 : total_ethanol = 30) :
  ∃ (fuel_A : ℝ), fuel_A = 82 ∧ 
  ethanol_A * fuel_A + ethanol_B * (tank_capacity - fuel_A) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l1886_188688


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l1886_188692

theorem quadratic_equation_range (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 4*x + m - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^2 - 4*x₁ + m - 1 = 0) →
  (x₂^2 - 4*x₂ + m - 1 = 0) →
  (3*x₁*x₂ - x₁ - x₂ > 2) →
  3 < m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l1886_188692


namespace NUMINAMATH_CALUDE_zongzi_sales_and_profit_l1886_188608

/-- The daily sales volume function for zongzi -/
def sales_volume (x : ℝ) : ℝ := 800 * x + 400

/-- The maximum daily production of zongzi -/
def max_production : ℝ := 1100

/-- The initial profit per zongzi in yuan -/
def initial_profit : ℝ := 2

/-- The total profit function for zongzi sales -/
def total_profit (x : ℝ) : ℝ := (initial_profit - x) * sales_volume x

theorem zongzi_sales_and_profit :
  (sales_volume 0.2 = 560) ∧ 
  (total_profit 0.2 = 1008) ∧
  (∃ x : ℝ, total_profit x = 1200 ∧ x = 0.5 ∧ sales_volume x ≤ max_production) :=
by sorry

end NUMINAMATH_CALUDE_zongzi_sales_and_profit_l1886_188608


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1886_188603

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem plywood_cut_perimeter_difference :
  let plywood := Rectangle.mk 12 6
  let area := plywood.length * plywood.width
  ∀ (piece : Rectangle),
    (6 * piece.length * piece.width = area) →
    (∃ (max_piece min_piece : Rectangle),
      (6 * max_piece.length * max_piece.width = area) ∧
      (6 * min_piece.length * min_piece.width = area) ∧
      (∀ (r : Rectangle), (6 * r.length * r.width = area) →
        perimeter r ≤ perimeter max_piece ∧
        perimeter r ≥ perimeter min_piece)) →
    (perimeter max_piece - perimeter min_piece = 14) := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1886_188603


namespace NUMINAMATH_CALUDE_cubic_function_c_value_l1886_188620

theorem cubic_function_c_value (a b c d y₁ y₂ : ℝ) :
  y₁ = a + b + c + d →
  y₂ = 8*a + 4*b + 2*c + d →
  y₁ - y₂ = -17 →
  c = -17 + 7*a + 3*b :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_c_value_l1886_188620


namespace NUMINAMATH_CALUDE_olivia_money_made_l1886_188681

/-- Represents the types of chocolate bars -/
inductive ChocolateType
| A
| B
| C

/-- The cost of each type of chocolate bar -/
def cost (t : ChocolateType) : ℕ :=
  match t with
  | ChocolateType.A => 3
  | ChocolateType.B => 4
  | ChocolateType.C => 5

/-- The total number of bars in the box -/
def total_bars : ℕ := 15

/-- The number of bars of each type in the box -/
def bars_in_box (t : ChocolateType) : ℕ :=
  match t with
  | ChocolateType.A => 7
  | ChocolateType.B => 5
  | ChocolateType.C => 3

/-- The number of bars sold of each type -/
def bars_sold (t : ChocolateType) : ℕ :=
  match t with
  | ChocolateType.A => 4
  | ChocolateType.B => 3
  | ChocolateType.C => 2

/-- The total money made from selling the chocolate bars -/
def total_money : ℕ :=
  (bars_sold ChocolateType.A * cost ChocolateType.A) +
  (bars_sold ChocolateType.B * cost ChocolateType.B) +
  (bars_sold ChocolateType.C * cost ChocolateType.C)

theorem olivia_money_made :
  total_money = 34 :=
by sorry

end NUMINAMATH_CALUDE_olivia_money_made_l1886_188681
