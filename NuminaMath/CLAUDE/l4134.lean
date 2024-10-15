import Mathlib

namespace NUMINAMATH_CALUDE_no_fractional_solution_l4134_413406

theorem no_fractional_solution (x y : ℝ) : 
  (∃ m n : ℤ, 13 * x + 4 * y = m ∧ 10 * x + 3 * y = n) → 
  (∃ a b : ℤ, x = a ∧ y = b) :=
by sorry

end NUMINAMATH_CALUDE_no_fractional_solution_l4134_413406


namespace NUMINAMATH_CALUDE_parallelogram_angle_measure_l4134_413496

theorem parallelogram_angle_measure (a b : ℝ) : 
  a = 70 → b = a + 40 → b = 110 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_measure_l4134_413496


namespace NUMINAMATH_CALUDE_min_distance_to_line_l4134_413401

/-- The minimum distance from the origin to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∀ p ∈ line, Real.sqrt ((0 - p.1)^2 + (0 - p.2)^2) ≥ 2 * Real.sqrt 2 ∧
  ∃ q ∈ line, Real.sqrt ((0 - q.1)^2 + (0 - q.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l4134_413401


namespace NUMINAMATH_CALUDE_remainder_calculation_l4134_413410

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- Theorem statement
theorem remainder_calculation :
  rem (5/9 : ℚ) (-3/4 : ℚ) = -7/36 := by
  sorry

end NUMINAMATH_CALUDE_remainder_calculation_l4134_413410


namespace NUMINAMATH_CALUDE_simplified_expression_equals_result_l4134_413478

theorem simplified_expression_equals_result (a b : ℝ) 
  (ha : a = 4) (hb : b = 3) : 
  (a * Real.sqrt (1 / a) + Real.sqrt (4 * b)) - (Real.sqrt a / 2 - b * Real.sqrt (1 / b)) = 1 + 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_result_l4134_413478


namespace NUMINAMATH_CALUDE_veronica_cherry_pitting_time_l4134_413437

/-- Represents the time needed to pit cherries for a cherry pie --/
def cherry_pitting_time (pounds_needed : ℕ) 
                        (cherries_per_pound : ℕ) 
                        (first_pound_rate : ℚ) 
                        (second_pound_rate : ℚ) 
                        (third_pound_rate : ℚ) 
                        (interruptions : ℕ) 
                        (interruption_duration : ℚ) : ℚ :=
  sorry

theorem veronica_cherry_pitting_time :
  cherry_pitting_time 3 80 (10/20) (8/20) (12/20) 2 15 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_veronica_cherry_pitting_time_l4134_413437


namespace NUMINAMATH_CALUDE_gum_pack_size_l4134_413462

-- Define the number of cherry and grape gum pieces
def cherry_gum : ℚ := 25
def grape_gum : ℚ := 35

-- Define the number of packs of grape gum found
def grape_packs_found : ℚ := 6

-- Define the variable x as the number of pieces in a complete pack
variable (x : ℚ)

-- Define the equality condition
def equality_condition (x : ℚ) : Prop :=
  (cherry_gum - x) / grape_gum = cherry_gum / (grape_gum + grape_packs_found * x)

-- Theorem statement
theorem gum_pack_size :
  equality_condition x → x = 115 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_gum_pack_size_l4134_413462


namespace NUMINAMATH_CALUDE_total_space_after_compaction_l4134_413419

/-- Represents the types of cans -/
inductive CanType
  | Small
  | Large

/-- Represents the properties of a can type -/
structure CanProperties where
  originalSize : ℕ
  compactionRate : ℚ

/-- Calculates the space taken by a type of can after compaction -/
def spaceAfterCompaction (props : CanProperties) (quantity : ℕ) : ℚ :=
  ↑(props.originalSize * quantity) * props.compactionRate

theorem total_space_after_compaction :
  let smallCanProps : CanProperties := ⟨20, 3/10⟩
  let largeCanProps : CanProperties := ⟨40, 4/10⟩
  let smallCanQuantity : ℕ := 50
  let largeCanQuantity : ℕ := 50
  let totalSpaceAfterCompaction :=
    spaceAfterCompaction smallCanProps smallCanQuantity +
    spaceAfterCompaction largeCanProps largeCanQuantity
  totalSpaceAfterCompaction = 1100 := by
  sorry


end NUMINAMATH_CALUDE_total_space_after_compaction_l4134_413419


namespace NUMINAMATH_CALUDE_terminal_side_point_theorem_l4134_413421

theorem terminal_side_point_theorem (m : ℝ) (hm : m ≠ 0) :
  let α := Real.arctan (3 * m / (-4 * m))
  (2 * Real.sin α + Real.cos α = 2/5) ∨ (2 * Real.sin α + Real.cos α = -2/5) := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_point_theorem_l4134_413421


namespace NUMINAMATH_CALUDE_right_triangle_from_special_case_l4134_413497

/-- 
Given a triangle with sides a, 2a, and c, where the angle between sides a and 2a is 60°,
prove that the angle opposite side 2a is 90°.
-/
theorem right_triangle_from_special_case (a : ℝ) (h : a > 0) :
  let c := a * Real.sqrt 3
  let cos_alpha := (a^2 + c^2 - (2*a)^2) / (2 * a * c)
  cos_alpha = 0 := by sorry

end NUMINAMATH_CALUDE_right_triangle_from_special_case_l4134_413497


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4134_413488

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i ^ 2 = -1 →
  Complex.im (i / (2 + i)) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4134_413488


namespace NUMINAMATH_CALUDE_find_number_l4134_413468

theorem find_number (N : ℚ) : 
  (N / (4/5) = (4/5) * N + 36) → N = 80 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l4134_413468


namespace NUMINAMATH_CALUDE_age_when_dog_born_is_15_l4134_413466

/-- The age of the person when their dog was born -/
def age_when_dog_born (current_age : ℕ) (dog_future_age : ℕ) (years_until_future : ℕ) : ℕ :=
  current_age - (dog_future_age - years_until_future)

/-- Theorem stating the age when the dog was born -/
theorem age_when_dog_born_is_15 :
  age_when_dog_born 17 4 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_age_when_dog_born_is_15_l4134_413466


namespace NUMINAMATH_CALUDE_solution1_satisfies_system1_solution2_satisfies_system2_l4134_413474

-- Part 1
def system1 (y z : ℝ → ℝ) : Prop :=
  ∀ x, (deriv y x + 2 * y x - 4 * z x = 0) ∧
       (deriv z x + y x - 3 * z x = 3 * x^2)

def solution1 (C₁ C₂ : ℝ) (y z : ℝ → ℝ) : Prop :=
  ∀ x, y x = C₁ * Real.exp (-x) + C₂ * Real.exp (2*x) - 6*x^2 + 6*x - 9 ∧
       z x = (1/4) * C₁ * Real.exp (-x) + C₂ * Real.exp (2*x) - 3*x^2 - 3

theorem solution1_satisfies_system1 (C₁ C₂ : ℝ) :
  ∀ y z, solution1 C₁ C₂ y z → system1 y z := by sorry

-- Part 2
def system2 (u v w : ℝ → ℝ) : Prop :=
  ∀ x, (6 * deriv u x - u x - 7 * v x + 5 * w x = 10 * Real.exp x) ∧
       (2 * deriv v x + u x + v x - w x = 0) ∧
       (3 * deriv w x - u x + 2 * v x - w x = Real.exp x)

def solution2 (C₁ C₂ C₃ : ℝ) (u v w : ℝ → ℝ) : Prop :=
  ∀ x, u x = C₁ + C₂ * Real.cos x + C₃ * Real.sin x + Real.exp x ∧
       v x = 2*C₁ + (1/2)*(C₃ - C₂)*Real.cos x - (1/2)*(C₃ + C₂)*Real.sin x ∧
       w x = 3*C₁ - (1/2)*(C₂ + C₃)*Real.cos x + (1/2)*(C₂ - C₃)*Real.sin x + Real.exp x

theorem solution2_satisfies_system2 (C₁ C₂ C₃ : ℝ) :
  ∀ u v w, solution2 C₁ C₂ C₃ u v w → system2 u v w := by sorry

end NUMINAMATH_CALUDE_solution1_satisfies_system1_solution2_satisfies_system2_l4134_413474


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l4134_413476

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def number_with_d (d : ℕ) : ℕ := 563000 + d * 100 + 4

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by_9 (number_with_d d) ∧
    ∀ (d' : ℕ), d' < d → ¬(is_divisible_by_9 (number_with_d d')) :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l4134_413476


namespace NUMINAMATH_CALUDE_gcd_204_85_l4134_413407

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l4134_413407


namespace NUMINAMATH_CALUDE_uncoolParentsOnlyChildCount_l4134_413477

/-- Represents a class of students -/
structure PhysicsClass where
  total : ℕ
  coolDads : ℕ
  coolMoms : ℕ
  coolBothAndSiblings : ℕ

/-- Calculates the number of students with uncool parents and no siblings -/
def uncoolParentsOnlyChild (c : PhysicsClass) : ℕ :=
  c.total - (c.coolDads + c.coolMoms - c.coolBothAndSiblings)

/-- The theorem to be proved -/
theorem uncoolParentsOnlyChildCount (c : PhysicsClass) 
  (h1 : c.total = 40)
  (h2 : c.coolDads = 20)
  (h3 : c.coolMoms = 22)
  (h4 : c.coolBothAndSiblings = 10) :
  uncoolParentsOnlyChild c = 8 := by
  sorry

#eval uncoolParentsOnlyChild { total := 40, coolDads := 20, coolMoms := 22, coolBothAndSiblings := 10 }

end NUMINAMATH_CALUDE_uncoolParentsOnlyChildCount_l4134_413477


namespace NUMINAMATH_CALUDE_carolyn_piano_practice_time_l4134_413439

/-- Given Carolyn's practice schedule, prove she practices piano for 20 minutes daily. -/
theorem carolyn_piano_practice_time :
  ∀ (piano_time : ℕ),
    (∃ (violin_time : ℕ), violin_time = 3 * piano_time) →
    (∃ (weekly_practice : ℕ), weekly_practice = 6 * (piano_time + 3 * piano_time)) →
    (∃ (monthly_practice : ℕ), monthly_practice = 4 * 6 * (piano_time + 3 * piano_time)) →
    4 * 6 * (piano_time + 3 * piano_time) = 1920 →
    piano_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_piano_practice_time_l4134_413439


namespace NUMINAMATH_CALUDE_expected_pairs_in_both_arrangements_l4134_413411

/-- Represents a 7x7 grid arrangement of numbers 1 through 49 -/
def Arrangement := Fin 49 → Fin 7 × Fin 7

/-- The number of rows in the grid -/
def num_rows : Nat := 7

/-- The number of columns in the grid -/
def num_cols : Nat := 7

/-- The total number of numbers in the grid -/
def total_numbers : Nat := num_rows * num_cols

/-- Calculates the expected number of pairs that occur in the same row or column in both arrangements -/
noncomputable def expected_pairs (a1 a2 : Arrangement) : ℝ :=
  (total_numbers.choose 2 : ℝ) * (1 / 16)

/-- The main theorem stating the expected number of pairs -/
theorem expected_pairs_in_both_arrangements :
  ∀ a1 a2 : Arrangement, expected_pairs a1 a2 = 73.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_pairs_in_both_arrangements_l4134_413411


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l4134_413465

theorem ellipse_equation_proof (a b : ℝ) : 
  (∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 2 ∧ y = 0) → -- ellipse passes through (2, 0)
  (a^2 - b^2 = 2) → -- ellipse shares focus with hyperbola x² - y² = 1
  (a^2 = 4 ∧ b^2 = 2) → -- derived from the conditions
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_equation_proof_l4134_413465


namespace NUMINAMATH_CALUDE_pyramid_section_volume_l4134_413473

/-- Given a pyramid with base area 3 and volume 3, and two parallel cross-sections with areas 1 and 2,
    the volume of the part of the pyramid between these cross-sections is (2√6 - √3) / 3. -/
theorem pyramid_section_volume 
  (base_area : ℝ) 
  (pyramid_volume : ℝ) 
  (section_area_1 : ℝ) 
  (section_area_2 : ℝ) 
  (h_base_area : base_area = 3) 
  (h_pyramid_volume : pyramid_volume = 3) 
  (h_section_area_1 : section_area_1 = 1) 
  (h_section_area_2 : section_area_2 = 2) : 
  ∃ (section_volume : ℝ), section_volume = (2 * Real.sqrt 6 - Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_section_volume_l4134_413473


namespace NUMINAMATH_CALUDE_telephone_number_D_is_9_l4134_413416

def TelephoneNumber (A B C D E F G H I J : ℕ) : Prop :=
  A > B ∧ B > C ∧ D > E ∧ E > F ∧ G > H ∧ H > I ∧ I > J ∧
  A % 2 = 0 ∧ B = A - 2 ∧ C = B - 2 ∧
  D % 2 = 1 ∧ E = D - 2 ∧ F = E - 2 ∧
  H + I + J = 9 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J

theorem telephone_number_D_is_9 :
  ∀ A B C D E F G H I J, TelephoneNumber A B C D E F G H I J → D = 9 :=
by sorry

end NUMINAMATH_CALUDE_telephone_number_D_is_9_l4134_413416


namespace NUMINAMATH_CALUDE_ted_losses_l4134_413432

/-- Represents a player in the game --/
inductive Player
| Carl
| James
| Saif
| Ted

/-- Records the number of wins and losses for a player --/
structure PlayerRecord where
  wins : Nat
  losses : Nat

/-- Represents the game results for all players --/
def GameResults := Player → PlayerRecord

theorem ted_losses (results : GameResults) :
  (results Player.Carl).wins = 5 ∧
  (results Player.Carl).losses = 0 ∧
  (results Player.James).wins = 4 ∧
  (results Player.James).losses = 2 ∧
  (results Player.Saif).wins = 1 ∧
  (results Player.Saif).losses = 6 ∧
  (results Player.Ted).wins = 4 ∧
  (∀ p : Player, (results p).wins + (results p).losses = 
    (results Player.Carl).wins + (results Player.James).wins + 
    (results Player.Saif).wins + (results Player.Ted).wins) →
  (results Player.Ted).losses = 6 := by
  sorry

end NUMINAMATH_CALUDE_ted_losses_l4134_413432


namespace NUMINAMATH_CALUDE_pasture_rental_problem_l4134_413460

/-- Calculate the rent share for a person based on their oxen usage and total pasture usage -/
def calculate_rent_share (total_rent : ℚ) (person_oxen_months : ℕ) (total_oxen_months : ℕ) : ℚ :=
  (person_oxen_months : ℚ) * total_rent / (total_oxen_months : ℚ)

/-- Represents the pasture rental problem -/
theorem pasture_rental_problem :
  let total_rent : ℚ := 750
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let c_months := 3
  let d_oxen := 18
  let d_months := 6
  let e_oxen := 20
  let e_months := 4
  let f_oxen := 25
  let f_months := 2
  let total_oxen_months := a_oxen * a_months + b_oxen * b_months + c_oxen * c_months +
                           d_oxen * d_months + e_oxen * e_months + f_oxen * f_months
  let c_share := calculate_rent_share total_rent (c_oxen * c_months) total_oxen_months
  c_share = 81.75 := by
    sorry

end NUMINAMATH_CALUDE_pasture_rental_problem_l4134_413460


namespace NUMINAMATH_CALUDE_extremum_implies_f_two_l4134_413452

/-- A cubic function with integer coefficients -/
def f (a b : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- Theorem stating that if f has an extremum of 10 at x = 1, then f(2) = 2 -/
theorem extremum_implies_f_two (a b : ℤ) :
  (f a b 1 = 10) →  -- f(1) = 10
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) →  -- local maximum at x = 1
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≥ f a b 1) →  -- local minimum at x = 1
  f a b 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_f_two_l4134_413452


namespace NUMINAMATH_CALUDE_car_speed_problem_l4134_413499

/-- Proves that given the conditions of the car problem, the speed of Car B is 50 km/h -/
theorem car_speed_problem (speed_b : ℝ) : 
  let speed_a := 3 * speed_b
  let time_a := 6
  let time_b := 2
  let total_distance := 1000
  speed_a * time_a + speed_b * time_b = total_distance →
  speed_b = 50 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l4134_413499


namespace NUMINAMATH_CALUDE_f_evaluation_l4134_413422

/-- The function f(x) = 3x^2 - 5x + 8 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

/-- Theorem stating that 3f(4) + 2f(-4) = 260 -/
theorem f_evaluation : 3 * f 4 + 2 * f (-4) = 260 := by
  sorry

end NUMINAMATH_CALUDE_f_evaluation_l4134_413422


namespace NUMINAMATH_CALUDE_pyramid_edges_l4134_413484

/-- Represents a pyramid with a polygonal base -/
structure Pyramid where
  base_sides : ℕ

/-- The number of vertices in a pyramid -/
def num_vertices (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of faces in a pyramid -/
def num_faces (p : Pyramid) : ℕ := p.base_sides + 1

/-- The number of edges in a pyramid -/
def num_edges (p : Pyramid) : ℕ := p.base_sides + p.base_sides

theorem pyramid_edges (p : Pyramid) :
  num_vertices p + num_faces p = 16 → num_edges p = 14 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edges_l4134_413484


namespace NUMINAMATH_CALUDE_harmonic_mean_three_fourths_five_sixths_l4134_413480

def harmonic_mean (a b : ℚ) : ℚ := 2 / (1/a + 1/b)

theorem harmonic_mean_three_fourths_five_sixths :
  harmonic_mean (3/4) (5/6) = 15/19 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_three_fourths_five_sixths_l4134_413480


namespace NUMINAMATH_CALUDE_greatest_multiple_5_and_6_less_than_800_l4134_413434

theorem greatest_multiple_5_and_6_less_than_800 : 
  ∃ n : ℕ, n = 780 ∧ 
  (∀ m : ℕ, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 800 → m ≤ n) ∧
  n % 5 = 0 ∧ n % 6 = 0 ∧ n < 800 :=
sorry

end NUMINAMATH_CALUDE_greatest_multiple_5_and_6_less_than_800_l4134_413434


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_y_eq_one_seventh_l4134_413450

theorem matrix_not_invertible_iff_y_eq_one_seventh :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2 + y, 5; 4 - y, 9]
  ¬(IsUnit (Matrix.det A)) ↔ y = (1 : ℝ) / 7 :=
by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_y_eq_one_seventh_l4134_413450


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l4134_413451

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Pay per task in dollars -/
def pay_per_task : ℚ := 6/5

/-- Number of working days per week -/
def working_days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * pay_per_task * working_days_per_week

theorem tim_weekly_earnings : weekly_earnings = 720 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l4134_413451


namespace NUMINAMATH_CALUDE_rational_quadratic_integer_solutions_l4134_413482

theorem rational_quadratic_integer_solutions (r : ℚ) :
  (∃ x : ℤ, r * x^2 + (r + 1) * x + r = 1) ↔ (r = 1 ∨ r = -1/7) := by
  sorry

end NUMINAMATH_CALUDE_rational_quadratic_integer_solutions_l4134_413482


namespace NUMINAMATH_CALUDE_inequality_proof_l4134_413442

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((2*a + b + c)^2 / (2*a^2 + (b + c)^2)) + 
  ((2*b + c + a)^2 / (2*b^2 + (c + a)^2)) + 
  ((2*c + a + b)^2 / (2*c^2 + (a + b)^2)) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4134_413442


namespace NUMINAMATH_CALUDE_cricket_game_initial_overs_l4134_413471

/-- Proves that the number of overs played initially is 10 in a cricket game scenario -/
theorem cricket_game_initial_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : target = 282) (h2 : initial_rate = 3.2) 
  (h3 : required_rate = 6.25) (h4 : remaining_overs = 40) : 
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
  initial_rate * initial_overs + required_rate * remaining_overs = target :=
by sorry

end NUMINAMATH_CALUDE_cricket_game_initial_overs_l4134_413471


namespace NUMINAMATH_CALUDE_calculation_proof_l4134_413459

theorem calculation_proof : -1^4 + 16 / (-2)^3 * |(-3) - 1| = -9 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4134_413459


namespace NUMINAMATH_CALUDE_value_of_a_l4134_413443

theorem value_of_a (a b c : ℝ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 6) 
  (eq3 : c = 4) : 
  a = 2 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l4134_413443


namespace NUMINAMATH_CALUDE_point_translation_l4134_413440

def initial_point : ℝ × ℝ := (-2, 3)

def translate_down (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - units)

def translate_right (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 + units, p.2)

theorem point_translation :
  (translate_right (translate_down initial_point 3) 1) = (-1, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_translation_l4134_413440


namespace NUMINAMATH_CALUDE_circle_max_sum_of_abs_l4134_413444

theorem circle_max_sum_of_abs (x y : ℝ) :
  x^2 + y^2 = 4 → |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_sum_of_abs_l4134_413444


namespace NUMINAMATH_CALUDE_platform_length_l4134_413470

/-- Given a train of length 300 meters, which takes 39 seconds to cross a platform
    and 9 seconds to cross a signal pole, the length of the platform is 1000 meters. -/
theorem platform_length
  (train_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 39)
  (h3 : pole_crossing_time = 9) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 1000 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l4134_413470


namespace NUMINAMATH_CALUDE_compound_simple_interest_principal_l4134_413418

theorem compound_simple_interest_principal (P r : ℝ) : 
  P * (1 + r)^2 - P = 11730 → P * r * 2 = 10200 → P = 17000 := by
  sorry

end NUMINAMATH_CALUDE_compound_simple_interest_principal_l4134_413418


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_and_twelve_l4134_413428

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ≠ 0 → d ≠ 10 → d < 10 → (n % 10 = d ∨ (n / 10) % 10 = d ∨ n / 100 = d) → n % d = 0

theorem largest_three_digit_divisible_by_digits_and_twelve :
  ∀ n : ℕ, is_three_digit n → divisible_by_digits n → n % 12 = 0 → n ≤ 864 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_and_twelve_l4134_413428


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l4134_413417

/-- A random variable following a normal distribution with mean 1 and standard deviation σ > 0 -/
def normal_rv (σ : ℝ) : Type := ℝ

/-- The probability density function of the normal distribution -/
noncomputable def pdf (σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The probability that the random variable takes a value in the interval (a, b) -/
noncomputable def prob (σ : ℝ) (a b : ℝ) : ℝ := sorry

/-- Theorem: If P(0 < ξ < 1) = 0.4, then P(0 < ξ < 2) = 0.8 for a normal distribution with mean 1 -/
theorem normal_distribution_probability (σ : ℝ) (hσ : σ > 0) :
  prob σ 0 1 = 0.4 → prob σ 0 2 = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l4134_413417


namespace NUMINAMATH_CALUDE_steak_entree_cost_l4134_413481

theorem steak_entree_cost 
  (total_guests : ℕ) 
  (chicken_cost : ℕ) 
  (total_budget : ℕ) 
  (h1 : total_guests = 80)
  (h2 : chicken_cost = 18)
  (h3 : total_budget = 1860)
  : (total_budget - (total_guests / 4 * chicken_cost)) / (3 * total_guests / 4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_steak_entree_cost_l4134_413481


namespace NUMINAMATH_CALUDE_three_A_students_l4134_413464

-- Define the students
inductive Student : Type
| Edward : Student
| Fiona : Student
| George : Student
| Hannah : Student
| Ian : Student

-- Define a predicate for getting an A
def got_A : Student → Prop := sorry

-- Define the statements
axiom Edward_statement : got_A Student.Edward → got_A Student.Fiona
axiom Fiona_statement : got_A Student.Fiona → got_A Student.George
axiom George_statement : got_A Student.George → got_A Student.Hannah
axiom Hannah_statement : got_A Student.Hannah → got_A Student.Ian

-- Define the condition that exactly three students got an A
axiom three_A : ∃ (s1 s2 s3 : Student), 
  (s1 ≠ s2 ∧ s1 ≠ s3 ∧ s2 ≠ s3) ∧
  got_A s1 ∧ got_A s2 ∧ got_A s3 ∧
  (∀ (s : Student), got_A s → (s = s1 ∨ s = s2 ∨ s = s3))

-- The theorem to prove
theorem three_A_students : 
  got_A Student.George ∧ got_A Student.Hannah ∧ got_A Student.Ian ∧
  ¬got_A Student.Edward ∧ ¬got_A Student.Fiona :=
sorry

end NUMINAMATH_CALUDE_three_A_students_l4134_413464


namespace NUMINAMATH_CALUDE_paul_gave_35_books_l4134_413495

/-- The number of books Paul gave to his friend -/
def books_given_to_friend (initial_books sold_books remaining_books : ℕ) : ℕ :=
  initial_books - sold_books - remaining_books

/-- Theorem stating that Paul gave 35 books to his friend -/
theorem paul_gave_35_books : books_given_to_friend 108 11 62 = 35 := by
  sorry

end NUMINAMATH_CALUDE_paul_gave_35_books_l4134_413495


namespace NUMINAMATH_CALUDE_actual_car_body_mass_l4134_413461

/-- Represents the scale factor between the model and the actual car body -/
def scaleFactor : ℝ := 10

/-- Represents the mass of the model car body in kilograms -/
def modelMass : ℝ := 1

/-- Calculates the volume ratio between the actual car body and the model -/
def volumeRatio : ℝ := scaleFactor ^ 3

/-- Calculates the mass of the actual car body in kilograms -/
def actualMass : ℝ := modelMass * volumeRatio

/-- Theorem stating that the mass of the actual car body is 1000 kg -/
theorem actual_car_body_mass : actualMass = 1000 := by
  sorry

end NUMINAMATH_CALUDE_actual_car_body_mass_l4134_413461


namespace NUMINAMATH_CALUDE_floor_sqrt_27_squared_l4134_413455

theorem floor_sqrt_27_squared : ⌊Real.sqrt 27⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_27_squared_l4134_413455


namespace NUMINAMATH_CALUDE_absolute_value_equality_implies_product_zero_l4134_413447

theorem absolute_value_equality_implies_product_zero (x y : ℝ) :
  |x - Real.log y| = x + Real.log y → x * (y - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_implies_product_zero_l4134_413447


namespace NUMINAMATH_CALUDE_expression_simplification_l4134_413408

theorem expression_simplification (x : ℝ) :
  2*x*(4*x^2 - 3) - 4*(x^2 - 3*x + 8) = 8*x^3 - 4*x^2 + 6*x - 32 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4134_413408


namespace NUMINAMATH_CALUDE_nathan_ate_four_boxes_l4134_413454

def gumballs_per_box : ℕ := 5
def gumballs_eaten : ℕ := 20

theorem nathan_ate_four_boxes : 
  gumballs_eaten / gumballs_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_nathan_ate_four_boxes_l4134_413454


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l4134_413425

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the total cost of insulation for a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

theorem tank_insulation_cost :
  let l : ℝ := 6
  let w : ℝ := 3
  let h : ℝ := 2
  let costPerSqFt : ℝ := 20
  insulationCost l w h costPerSqFt = 1440 := by
  sorry

#eval insulationCost 6 3 2 20

end NUMINAMATH_CALUDE_tank_insulation_cost_l4134_413425


namespace NUMINAMATH_CALUDE_planes_perpendicular_from_line_l4134_413424

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_from_line (a : Line) (M N : Plane) :
  perpendicular a M → parallel a N → perpendicularPlanes N M :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_from_line_l4134_413424


namespace NUMINAMATH_CALUDE_parabola_equation_l4134_413449

/-- A parabola with the origin as vertex, coordinate axes as axes of symmetry, 
    and passing through the point (6, 4) has the equation y² = 8/3 * x or x² = 9 * y -/
theorem parabola_equation : ∃ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ (y^2 = 8/3 * x ∨ x^2 = 9 * y)) ∧
  f 0 = 0 ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  f 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l4134_413449


namespace NUMINAMATH_CALUDE_count_pairs_satisfying_inequality_l4134_413409

theorem count_pairs_satisfying_inequality : 
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 * p.2 < 30 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 6) (Finset.range 30))).card = 41 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_satisfying_inequality_l4134_413409


namespace NUMINAMATH_CALUDE_g_composition_equals_1200_l4134_413467

def g (x : ℝ) : ℝ := 7 * x + 3

theorem g_composition_equals_1200 : g (g (g 3)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_1200_l4134_413467


namespace NUMINAMATH_CALUDE_amy_remaining_money_l4134_413414

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial_amount : ℕ) (num_items : ℕ) (item_cost : ℕ) : ℕ :=
  initial_amount - (num_items * item_cost)

/-- Proves that Amy has $97 left after her purchase -/
theorem amy_remaining_money :
  remaining_money 100 3 1 = 97 := by
  sorry

end NUMINAMATH_CALUDE_amy_remaining_money_l4134_413414


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l4134_413436

def team_size : ℕ := 12
def center_players : ℕ := 2
def lineup_size : ℕ := 4

theorem starting_lineup_combinations :
  (center_players) * (team_size - 1) * (team_size - 2) * (team_size - 3) = 1980 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l4134_413436


namespace NUMINAMATH_CALUDE_order_of_roots_l4134_413446

theorem order_of_roots (a b c : ℝ) (ha : a = 2^(4/3)) (hb : b = 3^(2/3)) (hc : c = 25^(1/3)) :
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_order_of_roots_l4134_413446


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l4134_413494

theorem consecutive_integers_product (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 812 → (x + 1)^2 - x = 813 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l4134_413494


namespace NUMINAMATH_CALUDE_angle_sum_proof_l4134_413426

theorem angle_sum_proof (a b : Real) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2)
  (h1 : 4 * (Real.cos a)^3 - 3 * (Real.cos b)^3 = 2)
  (h2 : 4 * Real.cos (2*a) + 3 * Real.cos (2*b) = 1) :
  2*a + b = π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l4134_413426


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4134_413498

theorem arithmetic_calculation : 12 / 4 - 3 - 16 + 4 * 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4134_413498


namespace NUMINAMATH_CALUDE_ratio_comparison_l4134_413487

theorem ratio_comparison : ∀ (a b : ℕ), 
  a = 6 ∧ b = 7 →
  ∃ (x : ℕ), x = 3 ∧
  (a - x : ℚ) / (b - x : ℚ) < 3 / 4 ∧
  ∀ (y : ℕ), y < x →
  (a - y : ℚ) / (b - y : ℚ) ≥ 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ratio_comparison_l4134_413487


namespace NUMINAMATH_CALUDE_percentage_equality_l4134_413491

theorem percentage_equality (x : ℝ) : (80 / 100 * 600 = 50 / 100 * x) → x = 960 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l4134_413491


namespace NUMINAMATH_CALUDE_marys_max_earnings_l4134_413486

/-- Calculates the maximum weekly earnings for Mary given her work conditions --/
def max_weekly_earnings (max_hours : ℕ) (regular_rate : ℚ) (overtime_rate : ℚ) (higher_overtime_rate : ℚ) : ℚ :=
  let regular_pay := regular_rate * 40
  let overtime_pay := overtime_rate * 10
  let higher_overtime_pay := higher_overtime_rate * 10
  regular_pay + overtime_pay + higher_overtime_pay

/-- Theorem stating that Mary's maximum weekly earnings are $675 --/
theorem marys_max_earnings :
  let max_hours : ℕ := 60
  let regular_rate : ℚ := 10
  let overtime_rate : ℚ := regular_rate * (1 + 1/4)
  let higher_overtime_rate : ℚ := regular_rate * (1 + 1/2)
  max_weekly_earnings max_hours regular_rate overtime_rate higher_overtime_rate = 675 := by
  sorry

end NUMINAMATH_CALUDE_marys_max_earnings_l4134_413486


namespace NUMINAMATH_CALUDE_total_pencils_l4134_413427

theorem total_pencils (boxes : ℕ) (pencils_per_box : ℕ) (h1 : boxes = 162) (h2 : pencils_per_box = 4) : 
  boxes * pencils_per_box = 648 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l4134_413427


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l4134_413457

/-- Represents the distances walked by Spencer -/
structure WalkDistances where
  total : ℝ
  libraryToPostOffice : ℝ
  postOfficeToHome : ℝ

/-- Calculates the distance from house to library -/
def distanceHouseToLibrary (w : WalkDistances) : ℝ :=
  w.total - w.libraryToPostOffice - w.postOfficeToHome

/-- Theorem stating that the distance from house to library is 0.3 miles -/
theorem spencer_walk_distance (w : WalkDistances) 
  (h_total : w.total = 0.8)
  (h_lib_post : w.libraryToPostOffice = 0.1)
  (h_post_home : w.postOfficeToHome = 0.4) : 
  distanceHouseToLibrary w = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l4134_413457


namespace NUMINAMATH_CALUDE_new_alcohol_concentration_l4134_413405

/-- Calculates the new alcohol concentration after adding water to an alcohol solution -/
theorem new_alcohol_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 3)
  (h2 : initial_concentration = 0.33)
  (h3 : added_water = 1)
  : (initial_volume * initial_concentration) / (initial_volume + added_water) = 0.2475 := by
  sorry

end NUMINAMATH_CALUDE_new_alcohol_concentration_l4134_413405


namespace NUMINAMATH_CALUDE_square_extension_theorem_l4134_413483

/-- A configuration of points derived from a unit square. -/
structure SquareExtension where
  /-- The unit square ABCD -/
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  /-- Extension point E on AB extended -/
  E : ℝ × ℝ
  /-- Extension point F on DA extended -/
  F : ℝ × ℝ
  /-- Point G on ray FC such that FG = FE -/
  G : ℝ × ℝ
  /-- Point H on ray FC such that FH = 1 -/
  H : ℝ × ℝ
  /-- Intersection of FE and line through G parallel to CE -/
  J : ℝ × ℝ
  /-- Intersection of FE and line through H parallel to CJ -/
  K : ℝ × ℝ

  /-- ABCD forms a unit square -/
  h_unit_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)
  /-- BE = 1 -/
  h_BE : E = (2, 0)
  /-- AF = 5/9 -/
  h_AF : F = (0, 5/9)
  /-- FG = FE -/
  h_FG_eq_FE : dist F G = dist F E
  /-- FH = 1 -/
  h_FH : dist F H = 1
  /-- G is on ray FC -/
  h_G_on_FC : ∃ t : ℝ, t > 0 ∧ G = F + t • (C - F)
  /-- H is on ray FC -/
  h_H_on_FC : ∃ t : ℝ, t > 0 ∧ H = F + t • (C - F)
  /-- Line through G is parallel to CE -/
  h_G_parallel_CE : (G.2 - J.2) / (G.1 - J.1) = (C.2 - E.2) / (C.1 - E.1)
  /-- Line through H is parallel to CJ -/
  h_H_parallel_CJ : (H.2 - K.2) / (H.1 - K.1) = (C.2 - J.2) / (C.1 - J.1)

/-- The main theorem stating that FK = 349/97 in the given configuration. -/
theorem square_extension_theorem (se : SquareExtension) : dist se.F se.K = 349/97 := by
  sorry

end NUMINAMATH_CALUDE_square_extension_theorem_l4134_413483


namespace NUMINAMATH_CALUDE_least_product_xy_l4134_413438

theorem least_product_xy (x y : ℕ+) (h : (x : ℚ)⁻¹ + (3 * y : ℚ)⁻¹ = (6 : ℚ)⁻¹) :
  (∀ a b : ℕ+, (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (6 : ℚ)⁻¹ → x * y ≤ a * b) ∧ x * y = 48 :=
sorry

end NUMINAMATH_CALUDE_least_product_xy_l4134_413438


namespace NUMINAMATH_CALUDE_max_profit_at_eight_days_max_profit_value_l4134_413420

/-- Profit function for fruit wholesaler --/
def profit (x : ℕ) : ℝ :=
  let initial_amount := 500
  let purchase_price := 40
  let base_selling_price := 60
  let daily_price_increase := 2
  let daily_loss := 10
  let daily_storage_cost := 40
  let selling_price := base_selling_price + daily_price_increase * x
  let remaining_amount := initial_amount - daily_loss * x
  (selling_price * remaining_amount) - (daily_storage_cost * x) - (initial_amount * purchase_price)

/-- Maximum storage time in days --/
def max_storage_time : ℕ := 8

/-- Theorem: Maximum profit is achieved at 8 days of storage --/
theorem max_profit_at_eight_days :
  ∀ x : ℕ, x ≤ max_storage_time → profit x ≤ profit max_storage_time :=
sorry

/-- Theorem: Maximum profit is 11600 yuan --/
theorem max_profit_value :
  profit max_storage_time = 11600 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_eight_days_max_profit_value_l4134_413420


namespace NUMINAMATH_CALUDE_cone_radius_l4134_413413

/-- Given a cone with angle π/3 between generatrix and base, and volume 3π, its base radius is √3 -/
theorem cone_radius (angle : Real) (volume : Real) (radius : Real) : 
  angle = π / 3 → volume = 3 * π → 
  (1 / 3) * π * radius^2 * (radius * Real.sqrt 3) = volume → 
  radius = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_radius_l4134_413413


namespace NUMINAMATH_CALUDE_tan_150_degrees_l4134_413435

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l4134_413435


namespace NUMINAMATH_CALUDE_bills_omelet_preparation_time_l4134_413441

/-- Represents the time in minutes for various tasks in omelet preparation --/
structure OmeletPreparationTime where
  chop_pepper : ℕ
  chop_onion : ℕ
  grate_cheese : ℕ
  assemble_and_cook : ℕ

/-- Represents the quantities of ingredients and omelets --/
structure OmeletQuantities where
  peppers : ℕ
  onions : ℕ
  omelets : ℕ

/-- Calculates the total time for omelet preparation given preparation times and quantities --/
def total_preparation_time (prep_time : OmeletPreparationTime) (quantities : OmeletQuantities) : ℕ :=
  prep_time.chop_pepper * quantities.peppers +
  prep_time.chop_onion * quantities.onions +
  prep_time.grate_cheese * quantities.omelets +
  prep_time.assemble_and_cook * quantities.omelets

/-- Theorem stating that Bill's total preparation time for five omelets is 50 minutes --/
theorem bills_omelet_preparation_time :
  let prep_time : OmeletPreparationTime := {
    chop_pepper := 3,
    chop_onion := 4,
    grate_cheese := 1,
    assemble_and_cook := 5
  }
  let quantities : OmeletQuantities := {
    peppers := 4,
    onions := 2,
    omelets := 5
  }
  total_preparation_time prep_time quantities = 50 := by
  sorry

end NUMINAMATH_CALUDE_bills_omelet_preparation_time_l4134_413441


namespace NUMINAMATH_CALUDE_totient_product_inequality_l4134_413430

theorem totient_product_inequality (m n : ℕ) (h : m ≠ n) : 
  n * (Nat.totient n) ≠ m * (Nat.totient m) := by
  sorry

end NUMINAMATH_CALUDE_totient_product_inequality_l4134_413430


namespace NUMINAMATH_CALUDE_inequality_lower_bound_l4134_413423

theorem inequality_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 2*y) * (2/x + 1/y) ≥ 8 ∧
  ∀ ε > 0, ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (x₀ + 2*y₀) * (2/x₀ + 1/y₀) < 8 + ε :=
sorry

end NUMINAMATH_CALUDE_inequality_lower_bound_l4134_413423


namespace NUMINAMATH_CALUDE_parallel_chords_central_angles_l4134_413475

/-- Given a circle with parallel chords of lengths 5, 12, and 13 determining
    central angles α, β, and α + β radians respectively, where α + β < π,
    prove that α + β = π/2 -/
theorem parallel_chords_central_angles
  (α β : Real)
  (h1 : 0 < α) (h2 : 0 < β)
  (h3 : α + β < π)
  (h4 : 2 * Real.sin (α / 2) = 5 / (2 * R))
  (h5 : 2 * Real.sin (β / 2) = 12 / (2 * R))
  (h6 : 2 * Real.sin ((α + β) / 2) = 13 / (2 * R))
  (R : Real) (h7 : R > 0) :
  α + β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_chords_central_angles_l4134_413475


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l4134_413445

theorem geometric_sequence_product (a b c : ℝ) : 
  (8/3 < a) ∧ (a < b) ∧ (b < c) ∧ (c < 27/2) ∧ 
  (∃ q : ℝ, q ≠ 0 ∧ a = 8/3 * q ∧ b = 8/3 * q^2 ∧ c = 8/3 * q^3 ∧ 27/2 = 8/3 * q^4) →
  a * b * c = 216 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l4134_413445


namespace NUMINAMATH_CALUDE_mystery_number_l4134_413469

theorem mystery_number (x : ℕ) (h : x + 45 = 92) : x = 47 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_l4134_413469


namespace NUMINAMATH_CALUDE_recycling_points_l4134_413402

/-- The number of pounds needed to recycle to earn one point -/
def poundsPerPoint (gwenPounds friendsPounds totalPoints : ℕ) : ℚ :=
  (gwenPounds + friendsPounds : ℚ) / totalPoints

theorem recycling_points (gwenPounds friendsPounds totalPoints : ℕ) 
  (h1 : gwenPounds = 5)
  (h2 : friendsPounds = 13)
  (h3 : totalPoints = 6) :
  poundsPerPoint gwenPounds friendsPounds totalPoints = 3 := by
  sorry

end NUMINAMATH_CALUDE_recycling_points_l4134_413402


namespace NUMINAMATH_CALUDE_sequence_realignment_l4134_413489

def letter_cycle_length : ℕ := 6
def digit_cycle_length : ℕ := 4

theorem sequence_realignment :
  ∃ n : ℕ, n > 0 ∧ n % letter_cycle_length = 0 ∧ n % digit_cycle_length = 0 ∧
  ∀ m : ℕ, (m > 0 ∧ m % letter_cycle_length = 0 ∧ m % digit_cycle_length = 0) → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_realignment_l4134_413489


namespace NUMINAMATH_CALUDE_equation_solutions_l4134_413458

theorem equation_solutions : 
  let solutions : List ℂ := [
    4 + Complex.I * Real.sqrt 6,
    4 - Complex.I * Real.sqrt 6,
    4 + Complex.I * Real.sqrt (21 + Real.sqrt 433),
    4 - Complex.I * Real.sqrt (21 + Real.sqrt 433),
    4 + Complex.I * Real.sqrt (21 - Real.sqrt 433),
    4 - Complex.I * Real.sqrt (21 - Real.sqrt 433)
  ]
  ∀ x ∈ solutions, (x - 2)^6 + (x - 6)^6 = 32 ∧
  ∀ x : ℂ, (x - 2)^6 + (x - 6)^6 = 32 → x ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4134_413458


namespace NUMINAMATH_CALUDE_puppy_feeding_theorem_l4134_413493

/-- Given the number of puppies, portions of formula, and days, 
    calculates the number of times each puppy should be fed per day. -/
def feeding_frequency (puppies : ℕ) (portions : ℕ) (days : ℕ) : ℕ :=
  (portions / days) / puppies

/-- Proves that for 7 puppies, 105 portions, and 5 days, 
    the feeding frequency is 3 times per day. -/
theorem puppy_feeding_theorem :
  feeding_frequency 7 105 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_puppy_feeding_theorem_l4134_413493


namespace NUMINAMATH_CALUDE_simplify_expression_l4134_413472

theorem simplify_expression (x y : ℝ) : 3*y + 5*y + 6*y + 2*x + 4*x = 14*y + 6*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4134_413472


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l4134_413448

theorem adult_ticket_cost (child_cost : ℝ) : 
  (child_cost + 6 = 19) ∧ 
  (2 * (child_cost + 6) + 3 * child_cost = 77) := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l4134_413448


namespace NUMINAMATH_CALUDE_field_trip_cost_calculation_l4134_413400

def field_trip_cost (num_students : ℕ) (num_teachers : ℕ) (student_ticket_price : ℚ) 
  (adult_ticket_price : ℚ) (discount_rate : ℚ) (min_tickets_for_discount : ℕ) 
  (transportation_cost : ℚ) (meal_cost_per_person : ℚ) : ℚ :=
  sorry

theorem field_trip_cost_calculation : 
  field_trip_cost 25 6 1 3 0.2 20 100 7.5 = 366.9 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_cost_calculation_l4134_413400


namespace NUMINAMATH_CALUDE_pushups_total_l4134_413431

/-- The number of push-ups Zachary and David did altogether -/
def total_pushups (zachary_pushups : ℕ) (david_extra_pushups : ℕ) : ℕ :=
  zachary_pushups + (zachary_pushups + david_extra_pushups)

/-- Theorem stating that given the conditions, the total number of push-ups is 146 -/
theorem pushups_total : total_pushups 44 58 = 146 := by
  sorry

end NUMINAMATH_CALUDE_pushups_total_l4134_413431


namespace NUMINAMATH_CALUDE_triangle_properties_l4134_413415

/-- Given two 2D vectors a and b, proves statements about the triangle formed by 0, a, and b -/
theorem triangle_properties (a b : Fin 2 → ℝ) 
  (ha : a = ![4, -1]) 
  (hb : b = ![2, 6]) : 
  (1/2 * abs (a 0 * b 1 - a 1 * b 0) = 13) ∧ 
  ((((a 0 + b 0)/2)^2 + ((a 1 + b 1)/2)^2) = 15.25) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4134_413415


namespace NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l4134_413429

theorem quadratic_trinomial_minimum (a b : ℝ) (h1 : a > b)
  (h2 : ∀ x : ℝ, a * x^2 + 2*x + b ≥ 0)
  (h3 : ∃ x₀ : ℝ, a * x₀^2 + 2*x₀ + b = 0) :
  (∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2) ∧
  (∃ x : ℝ, (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_minimum_l4134_413429


namespace NUMINAMATH_CALUDE_pens_to_classmates_l4134_413456

/-- Represents the problem of calculating the fraction of remaining pens given to classmates. -/
theorem pens_to_classmates 
  (boxes : ℕ) 
  (pens_per_box : ℕ) 
  (friend_percentage : ℚ) 
  (pens_left : ℕ) 
  (h1 : boxes = 20) 
  (h2 : pens_per_box = 5) 
  (h3 : friend_percentage = 2/5) 
  (h4 : pens_left = 45) : 
  (boxes * pens_per_box - pens_left - (friend_percentage * (boxes * pens_per_box))) / 
  ((1 - friend_percentage) * (boxes * pens_per_box)) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_pens_to_classmates_l4134_413456


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l4134_413490

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : 
  A = 400 * Real.pi → d = 40 → A = Real.pi * (d / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l4134_413490


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l4134_413492

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l4134_413492


namespace NUMINAMATH_CALUDE_harry_snails_collection_l4134_413433

/-- Represents the number of sea stars Harry collected initially -/
def sea_stars : ℕ := 34

/-- Represents the number of seashells Harry collected initially -/
def seashells : ℕ := 21

/-- Represents the total number of items Harry had at the end of his walk -/
def total_items_left : ℕ := 59

/-- Represents the number of sea creatures Harry lost during his walk -/
def lost_sea_creatures : ℕ := 25

/-- Represents the number of snails Harry collected initially -/
def snails_collected : ℕ := total_items_left - (sea_stars + seashells - lost_sea_creatures)

theorem harry_snails_collection :
  snails_collected = 29 :=
sorry

end NUMINAMATH_CALUDE_harry_snails_collection_l4134_413433


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l4134_413463

theorem coefficient_x_squared_in_expansion : 
  let expansion := (fun x : ℝ => (x - x⁻¹)^6)
  ∃ (a b c : ℝ), ∀ x : ℝ, x ≠ 0 → 
    expansion x = a*x^3 + 15*x^2 + b*x + c + (x⁻¹ * (1 + x⁻¹ * (1 + x⁻¹ * (1)))) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l4134_413463


namespace NUMINAMATH_CALUDE_equation_solution_l4134_413403

theorem equation_solution : ∃! x : ℤ, 45 - (28 - (x - (15 - 19))) = 58 ∧ x = 37 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4134_413403


namespace NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l4134_413453

theorem cos_pi_4_plus_alpha (α : Real) 
  (h : Real.sin (π / 4 - α) = Real.sqrt 2 / 2) : 
  Real.cos (π / 4 + α) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l4134_413453


namespace NUMINAMATH_CALUDE_line_slope_proof_l4134_413485

theorem line_slope_proof : 
  let A : ℝ := Real.sin (30 * π / 180)
  let B : ℝ := Real.cos (150 * π / 180)
  let slope : ℝ := -A / B
  slope = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_proof_l4134_413485


namespace NUMINAMATH_CALUDE_key_dimension_in_polygon_division_l4134_413404

/-- Represents a polygon with a key dimension --/
structure Polygon where
  keyDimension : ℝ

/-- Represents a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Function to check if two polygons are congruent --/
def areCongruent (p1 p2 : Polygon) : Prop := sorry

/-- Function to check if polygons can form a square --/
def canFormSquare (p1 p2 : Polygon) (s : Square) : Prop := sorry

/-- Theorem stating the existence of a key dimension x = 4 in the polygons --/
theorem key_dimension_in_polygon_division (r : Rectangle) 
  (h1 : r.width = 12 ∧ r.height = 12) 
  (p1 p2 : Polygon) (s : Square)
  (h2 : areCongruent p1 p2)
  (h3 : canFormSquare p1 p2 s)
  (h4 : s.side^2 = r.width * r.height) :
  ∃ x : ℝ, x = 4 ∧ (p1.keyDimension = x ∨ p2.keyDimension = x) :=
sorry

end NUMINAMATH_CALUDE_key_dimension_in_polygon_division_l4134_413404


namespace NUMINAMATH_CALUDE_smallest_multiple_of_9_l4134_413479

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem smallest_multiple_of_9 :
  ∀ d : ℕ, d < 10 →
    (is_multiple_of_9 456786 ∧
     is_six_digit 456786 ∧
     456786 = 45678 * 10 + 6) ∧
    (∀ n : ℕ, is_six_digit n ∧ n < 456786 ∧ ∃ d' : ℕ, d' < 10 ∧ n = 45678 * 10 + d' →
      ¬is_multiple_of_9 n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_9_l4134_413479


namespace NUMINAMATH_CALUDE_sixteen_team_tournament_games_l4134_413412

/-- Calculates the number of games in a single-elimination tournament. -/
def num_games_in_tournament (num_teams : ℕ) : ℕ :=
  num_teams - 1

/-- Theorem: In a single-elimination tournament with 16 teams, 15 games are played to determine the winner. -/
theorem sixteen_team_tournament_games :
  num_games_in_tournament 16 = 15 := by
  sorry

#eval num_games_in_tournament 16  -- Should output 15

end NUMINAMATH_CALUDE_sixteen_team_tournament_games_l4134_413412
