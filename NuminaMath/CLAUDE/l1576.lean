import Mathlib

namespace NUMINAMATH_CALUDE_expected_mass_with_error_l1576_157622

/-- The expected mass of 100 metal disks with manufacturing errors -/
theorem expected_mass_with_error (
  nominal_diameter : ℝ)
  (perfect_disk_mass : ℝ)
  (radius_std_dev : ℝ)
  (disk_count : ℕ)
  (h1 : nominal_diameter = 1)
  (h2 : perfect_disk_mass = 100)
  (h3 : radius_std_dev = 0.01)
  (h4 : disk_count = 100) :
  ∃ (expected_mass : ℝ), 
    expected_mass = disk_count * perfect_disk_mass * (1 + 4 * (radius_std_dev / nominal_diameter)^2) ∧
    expected_mass = 10004 :=
by sorry

end NUMINAMATH_CALUDE_expected_mass_with_error_l1576_157622


namespace NUMINAMATH_CALUDE_hcf_lcm_problem_l1576_157693

theorem hcf_lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 396) (h3 : b = 220) : a = 36 := by
  sorry

end NUMINAMATH_CALUDE_hcf_lcm_problem_l1576_157693


namespace NUMINAMATH_CALUDE_michelle_gas_usage_l1576_157662

theorem michelle_gas_usage (start_gas end_gas : ℝ) (h1 : start_gas = 0.5) (h2 : end_gas = 0.17) :
  start_gas - end_gas = 0.33 := by
sorry

end NUMINAMATH_CALUDE_michelle_gas_usage_l1576_157662


namespace NUMINAMATH_CALUDE_equal_savings_l1576_157680

theorem equal_savings (your_initial : ℕ) (friend_initial : ℕ) (your_rate : ℕ) (friend_rate : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  friend_initial = 210 →
  your_rate = 7 →
  friend_rate = 5 →
  weeks = 25 →
  your_initial + your_rate * weeks = friend_initial + friend_rate * weeks :=
by sorry

end NUMINAMATH_CALUDE_equal_savings_l1576_157680


namespace NUMINAMATH_CALUDE_parabola_theorem_l1576_157607

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

/-- Line passing through (1,0) -/
structure Line where
  k : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y = k*(x-1)

/-- Point on the parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : para.eq x y

/-- Circle passing through three points -/
def circle_passes_through (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (x3-x1)*(x2-x1) + (y3-y1)*(y2-y1) = 0

/-- Main theorem -/
theorem parabola_theorem (para : Parabola) :
  (∀ l : Line, ∀ P Q : ParabolaPoint para,
    l.eq P.x P.y ∧ l.eq Q.x Q.y →
    circle_passes_through P.x P.y Q.x Q.y 0 0) →
  para.p = 1/2 ∧
  (∀ R : ℝ × ℝ,
    (∃ P Q : ParabolaPoint para,
      R.1 = P.x + Q.x - 1/4 ∧
      R.2 = P.y + Q.y) →
    R.2^2 = R.1 - 7/4) :=
sorry

end NUMINAMATH_CALUDE_parabola_theorem_l1576_157607


namespace NUMINAMATH_CALUDE_union_when_t_is_two_B_subset_A_iff_l1576_157633

-- Define sets A and B
def A (t : ℝ) : Set ℝ := {x | x^2 + (1-t)*x - t ≤ 0}
def B : Set ℝ := {x | |x-2| < 1}

-- Statement 1
theorem union_when_t_is_two :
  A 2 ∪ B = {x | -1 ≤ x ∧ x < 3} := by sorry

-- Statement 2
theorem B_subset_A_iff (t : ℝ) :
  B ⊆ A t ↔ t ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_when_t_is_two_B_subset_A_iff_l1576_157633


namespace NUMINAMATH_CALUDE_sum_of_squares_parity_l1576_157631

theorem sum_of_squares_parity (a b c : ℤ) (h : Odd (a + b + c)) :
  Odd (a^2 + b^2 - c^2 + 2*a*b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_parity_l1576_157631


namespace NUMINAMATH_CALUDE_dwarf_system_stabilizes_l1576_157642

-- Define the color of a dwarf's house
inductive Color
| Red
| White

-- Define the state of the dwarf system
structure DwarfSystem :=
  (houses : Fin 12 → Color)
  (friends : Fin 12 → Set (Fin 12))

-- Define a single step in the system
def step (sys : DwarfSystem) (i : Fin 12) : DwarfSystem := sorry

-- Define the relation between two states
def reaches (initial final : DwarfSystem) : Prop := sorry

-- Theorem statement
theorem dwarf_system_stabilizes (initial : DwarfSystem) :
  ∃ (final : DwarfSystem), reaches initial final ∧ ∀ i, step final i = final :=
sorry

end NUMINAMATH_CALUDE_dwarf_system_stabilizes_l1576_157642


namespace NUMINAMATH_CALUDE_not_cube_sum_l1576_157660

theorem not_cube_sum (a b : ℤ) : ¬ ∃ (k : ℤ), a^3 + b^3 + 4 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_not_cube_sum_l1576_157660


namespace NUMINAMATH_CALUDE_fourth_roll_six_prob_l1576_157601

/-- Represents a six-sided die --/
structure Die where
  prob_six : ℚ
  prob_other : ℚ
  sum_probs : prob_six + 5 * prob_other = 1

/-- The fair die --/
def fair_die : Die where
  prob_six := 1/6
  prob_other := 1/6
  sum_probs := by norm_num

/-- The biased die --/
def biased_die : Die where
  prob_six := 3/4
  prob_other := 1/20
  sum_probs := by norm_num

/-- The probability of choosing each die --/
def prob_choose_die : ℚ := 1/2

/-- The number of initial rolls that are sixes --/
def num_initial_sixes : ℕ := 3

/-- Theorem: Given the conditions, the probability of rolling a six on the fourth roll is 2187/982 --/
theorem fourth_roll_six_prob :
  let prob_fair := prob_choose_die * fair_die.prob_six^num_initial_sixes
  let prob_biased := prob_choose_die * biased_die.prob_six^num_initial_sixes
  let total_prob := prob_fair + prob_biased
  let cond_prob_fair := prob_fair / total_prob
  let cond_prob_biased := prob_biased / total_prob
  cond_prob_fair * fair_die.prob_six + cond_prob_biased * biased_die.prob_six = 2187 / 982 := by
  sorry

end NUMINAMATH_CALUDE_fourth_roll_six_prob_l1576_157601


namespace NUMINAMATH_CALUDE_monday_greatest_range_l1576_157691

/-- Temperature range for a day -/
def temp_range (high low : Int) : Int := high - low

/-- Temperature data for each day -/
def monday_high : Int := 6
def monday_low : Int := -4
def tuesday_high : Int := 3
def tuesday_low : Int := -6
def wednesday_high : Int := 4
def wednesday_low : Int := -2
def thursday_high : Int := 4
def thursday_low : Int := -5
def friday_high : Int := 8
def friday_low : Int := 0

/-- Theorem: Monday has the greatest temperature range -/
theorem monday_greatest_range :
  let monday_range := temp_range monday_high monday_low
  let tuesday_range := temp_range tuesday_high tuesday_low
  let wednesday_range := temp_range wednesday_high wednesday_low
  let thursday_range := temp_range thursday_high thursday_low
  let friday_range := temp_range friday_high friday_low
  (monday_range > tuesday_range) ∧
  (monday_range > wednesday_range) ∧
  (monday_range > thursday_range) ∧
  (monday_range > friday_range) :=
by sorry

end NUMINAMATH_CALUDE_monday_greatest_range_l1576_157691


namespace NUMINAMATH_CALUDE_value_of_e_l1576_157648

theorem value_of_e : (14 : ℕ)^2 * 5^3 * 568 = 13916000 := by
  sorry

end NUMINAMATH_CALUDE_value_of_e_l1576_157648


namespace NUMINAMATH_CALUDE_cube_difference_l1576_157611

/-- Calculates the number of cubes needed for a hollow block -/
def hollow_block_cubes (length width depth : ℕ) : ℕ :=
  2 * length * width + 4 * (length + width) * (depth - 2) - 8 * (depth - 2)

/-- Calculates the number of cubes in a solid block -/
def solid_block_cubes (length width depth : ℕ) : ℕ :=
  length * width * depth

theorem cube_difference (length width depth : ℕ) 
  (h1 : length = 7)
  (h2 : width = 7)
  (h3 : depth = 6) : 
  solid_block_cubes length width depth - hollow_block_cubes length width depth = 100 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l1576_157611


namespace NUMINAMATH_CALUDE_no_positive_roots_l1576_157643

theorem no_positive_roots :
  ∀ x : ℝ, x > 0 → x^3 + 6*x^2 + 11*x + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_roots_l1576_157643


namespace NUMINAMATH_CALUDE_equation_equality_relationship_l1576_157649

-- Define what an equality is
def IsEquality (s : String) : Prop := true  -- All mathematical statements of the form a = b are equalities

-- Define what an equation is
def IsEquation (s : String) : Prop := IsEquality s ∧ ∃ x, s.contains x  -- An equation is an equality that contains unknowns

-- The statement we want to prove false
def statement : Prop :=
  (∀ s, IsEquation s → IsEquality s) ∧ (∀ s, IsEquality s → IsEquation s)

-- Theorem: The statement is false
theorem equation_equality_relationship : ¬statement := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_relationship_l1576_157649


namespace NUMINAMATH_CALUDE_sum_interior_angles_specific_polyhedron_l1576_157699

/-- A convex polyhedron with given number of vertices and edges -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ

/-- The sum of interior angles of all faces of a convex polyhedron -/
def sum_interior_angles (p : ConvexPolyhedron) : ℝ :=
  sorry

/-- Theorem stating the sum of interior angles for a specific convex polyhedron -/
theorem sum_interior_angles_specific_polyhedron :
  let p : ConvexPolyhedron := ⟨20, 30⟩
  sum_interior_angles p = 6480 :=
by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_specific_polyhedron_l1576_157699


namespace NUMINAMATH_CALUDE_gym_class_size_l1576_157612

/-- The number of students in the third group -/
def third_group_size : ℕ := 37

/-- The percentage of students in the third group -/
def third_group_percentage : ℚ := 1/2

/-- The total number of students in the gym class -/
def total_students : ℕ := 74

theorem gym_class_size :
  (third_group_size : ℚ) / third_group_percentage = total_students := by
  sorry

end NUMINAMATH_CALUDE_gym_class_size_l1576_157612


namespace NUMINAMATH_CALUDE_garden_area_l1576_157655

/-- The area of a garden with square cutouts -/
theorem garden_area (garden_length : ℝ) (garden_width : ℝ) 
  (cutout1_side : ℝ) (cutout2_side : ℝ) : 
  garden_length = 20 ∧ garden_width = 18 ∧ 
  cutout1_side = 4 ∧ cutout2_side = 5 →
  garden_length * garden_width - cutout1_side^2 - cutout2_side^2 = 319 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l1576_157655


namespace NUMINAMATH_CALUDE_data_transmission_time_data_transmission_problem_l1576_157687

theorem data_transmission_time : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (blocks : ℝ) (chunks_per_block : ℝ) (chunks_per_second : ℝ) (time_in_minutes : ℝ) =>
    blocks * chunks_per_block / chunks_per_second / 60 = time_in_minutes

theorem data_transmission_problem :
  data_transmission_time 100 600 150 7 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_data_transmission_problem_l1576_157687


namespace NUMINAMATH_CALUDE_pauls_recycling_bags_l1576_157678

theorem pauls_recycling_bags (x : ℕ) : 
  (∃ (bags_on_sunday : ℕ), 
    bags_on_sunday = 3 ∧ 
    (∀ (cans_per_bag : ℕ), cans_per_bag = 8 → 
      (∀ (total_cans : ℕ), total_cans = 72 → 
        cans_per_bag * (x + bags_on_sunday) = total_cans))) → 
  x = 6 := by sorry

end NUMINAMATH_CALUDE_pauls_recycling_bags_l1576_157678


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_l1576_157689

/-- An equilateral triangle with side length 6 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 6

/-- The centroid of a triangle -/
structure Centroid (T : EquilateralTriangle) where

/-- The perpendicular from the centroid to a side of the triangle -/
def perpendicular (T : EquilateralTriangle) (C : Centroid T) : ℝ := sorry

/-- The theorem stating the sum of perpendiculars from the centroid equals 3√3 -/
theorem sum_of_perpendiculars (T : EquilateralTriangle) (C : Centroid T) :
  3 * (perpendicular T C) = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_perpendiculars_l1576_157689


namespace NUMINAMATH_CALUDE_softball_players_count_l1576_157654

theorem softball_players_count (cricket hockey football total : ℕ) 
  (h1 : cricket = 15)
  (h2 : hockey = 12)
  (h3 : football = 13)
  (h4 : total = 55) :
  total - (cricket + hockey + football) = 15 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_count_l1576_157654


namespace NUMINAMATH_CALUDE_integer_properties_l1576_157658

theorem integer_properties (m n k : ℕ) (hm : m > 0) (hn : n > 0) : 
  ∃ (a b : ℕ), 
    -- (m+n)^2 + (m-n)^2 is even
    ∃ (c : ℕ), (m + n)^2 + (m - n)^2 = 2 * c ∧
    -- ((m+n)^2 + (m-n)^2) / 2 can be expressed as the sum of squares of two positive integers
    ((m + n)^2 + (m - n)^2) / 2 = a^2 + b^2 ∧
    -- For any integer k, (2k+1)^2 - (2k-1)^2 is divisible by 8
    ∃ (d : ℕ), (2 * k + 1)^2 - (2 * k - 1)^2 = 8 * d :=
by sorry

end NUMINAMATH_CALUDE_integer_properties_l1576_157658


namespace NUMINAMATH_CALUDE_no_rational_roots_l1576_157645

theorem no_rational_roots : 
  ∀ (p q : ℤ), q ≠ 0 → 3 * (p / q)^4 - 4 * (p / q)^3 - 9 * (p / q)^2 + 10 * (p / q) + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l1576_157645


namespace NUMINAMATH_CALUDE_unique_cyclic_number_l1576_157668

def is_permutation (a b : Nat) : Prop := sorry

def is_six_digit (n : Nat) : Prop := 100000 ≤ n ∧ n < 1000000

theorem unique_cyclic_number : ∃! x : Nat, 
  is_six_digit x ∧ 
  is_six_digit (2*x) ∧ 
  is_six_digit (3*x) ∧ 
  is_six_digit (4*x) ∧ 
  is_six_digit (5*x) ∧ 
  is_six_digit (6*x) ∧
  is_permutation x (2*x) ∧ 
  is_permutation x (3*x) ∧ 
  is_permutation x (4*x) ∧ 
  is_permutation x (5*x) ∧ 
  is_permutation x (6*x) ∧
  x = 142857 :=
by sorry

end NUMINAMATH_CALUDE_unique_cyclic_number_l1576_157668


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1576_157619

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation 
  (l1 l2 l : Line2D) (P : Point2D) : 
  l1 = Line2D.mk 1 2 (-11) →
  l2 = Line2D.mk 2 1 (-10) →
  pointOnLine P l1 →
  pointOnLine P l2 →
  pointOnLine P l →
  perpendicularLines l l2 →
  l = Line2D.mk 1 (-2) 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1576_157619


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l1576_157657

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 2, m^2}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem subset_implies_m_values (m : ℝ) : 
  B m ⊆ A m → m = 0 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l1576_157657


namespace NUMINAMATH_CALUDE_recycling_problem_l1576_157618

/-- Recycling problem -/
theorem recycling_problem (pounds_per_point : ℕ) (gwen_pounds : ℕ) (total_points : ℕ) 
  (h1 : pounds_per_point = 3)
  (h2 : gwen_pounds = 5)
  (h3 : total_points = 6) :
  gwen_pounds / pounds_per_point + (total_points - gwen_pounds / pounds_per_point) * pounds_per_point = 15 :=
by sorry

end NUMINAMATH_CALUDE_recycling_problem_l1576_157618


namespace NUMINAMATH_CALUDE_max_value_constraint_l1576_157673

theorem max_value_constraint (w x y z : ℝ) (h : 9*w^2 + 4*x^2 + y^2 + 25*z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 201 ∧ 
  (∀ w' x' y' z' : ℝ, 9*w'^2 + 4*x'^2 + y'^2 + 25*z'^2 = 1 → 
    9*w' + 4*x' + 2*y' + 10*z' ≤ max) ∧
  (∃ w'' x'' y'' z'' : ℝ, 9*w''^2 + 4*x''^2 + y''^2 + 25*z''^2 = 1 ∧
    9*w'' + 4*x'' + 2*y'' + 10*z'' = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1576_157673


namespace NUMINAMATH_CALUDE_value_of_b_minus_d_squared_l1576_157604

theorem value_of_b_minus_d_squared 
  (a b c d : ℤ) 
  (eq1 : a - b - c + d = 13) 
  (eq2 : a + b - c - d = 3) : 
  (b - d)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_minus_d_squared_l1576_157604


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l1576_157665

theorem sphere_volume_ratio (r₁ r₂ r₃ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ = 2 * r₁) (h₃ : r₃ = 3 * r₁) :
  (4 / 3) * π * r₃^3 = 3 * ((4 / 3) * π * r₁^3 + (4 / 3) * π * r₂^3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l1576_157665


namespace NUMINAMATH_CALUDE_triangle_problem_l1576_157696

noncomputable def Triangle (a b c : ℝ) (A B C : ℝ) := True

theorem triangle_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c)
  (h_c : c = Real.sqrt 7)
  (h_area : 1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :
  C = π/3 ∧ a + b = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l1576_157696


namespace NUMINAMATH_CALUDE_cannot_tile_figure_l1576_157682

/-- A figure that can be colored such that each 1 × 3 strip covers exactly one colored cell. -/
structure ColoredFigure where
  colored_cells : ℕ

/-- A strip used for tiling. -/
structure Strip where
  width : ℕ
  height : ℕ

/-- Predicate to check if a figure can be tiled with given strips. -/
def CanBeTiled (f : ColoredFigure) (s : Strip) : Prop :=
  f.colored_cells % s.width = 0

theorem cannot_tile_figure (f : ColoredFigure) (s : Strip) 
  (h1 : f.colored_cells = 7)
  (h2 : s.width = 3)
  (h3 : s.height = 1) : 
  ¬CanBeTiled f s := by
  sorry

end NUMINAMATH_CALUDE_cannot_tile_figure_l1576_157682


namespace NUMINAMATH_CALUDE_cubic_monotone_implies_a_bound_l1576_157646

/-- A function f is monotonically increasing on an interval (a, b) if for any x, y in (a, b) with x < y, we have f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

/-- The cubic function f(x) = ax³ - x² + x - 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

theorem cubic_monotone_implies_a_bound :
  ∀ a : ℝ, MonotonicallyIncreasing (f a) 1 2 → a > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_monotone_implies_a_bound_l1576_157646


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_three_l1576_157669

theorem no_solution_iff_m_eq_neg_three (m : ℝ) :
  (∀ x : ℝ, x ≠ -1 → (3 * x) / (x + 1) ≠ m / (x + 1) + 2) ↔ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_three_l1576_157669


namespace NUMINAMATH_CALUDE_function_existence_iff_divisibility_l1576_157675

theorem function_existence_iff_divisibility (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[k] n = n + a)) ↔ (a ≥ 0 ∧ k ∣ a) :=
sorry

end NUMINAMATH_CALUDE_function_existence_iff_divisibility_l1576_157675


namespace NUMINAMATH_CALUDE_min_sum_consecutive_multiples_l1576_157636

theorem min_sum_consecutive_multiples : 
  ∃ (a b c d : ℕ), 
    (b = a + 1) ∧ 
    (c = b + 1) ∧ 
    (d = c + 1) ∧
    (∃ k : ℕ, a = 11 * k) ∧
    (∃ l : ℕ, b = 7 * l) ∧
    (∃ m : ℕ, c = 5 * m) ∧
    (∃ n : ℕ, d = 3 * n) ∧
    (∀ w x y z : ℕ, 
      (x = w + 1) ∧ 
      (y = x + 1) ∧ 
      (z = y + 1) ∧
      (∃ p : ℕ, w = 11 * p) ∧
      (∃ q : ℕ, x = 7 * q) ∧
      (∃ r : ℕ, y = 5 * r) ∧
      (∃ s : ℕ, z = 3 * s) →
      (a + b + c + d ≤ w + x + y + z)) ∧
    (a + b + c + d = 1458) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_consecutive_multiples_l1576_157636


namespace NUMINAMATH_CALUDE_base4_calculation_l1576_157640

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Multiplication in base 4 --/
def mulBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a * base4ToBase10 b)

/-- Division in base 4 --/
def divBase4 (a b : ℕ) : ℕ := base10ToBase4 (base4ToBase10 a / base4ToBase10 b)

theorem base4_calculation : 
  mulBase4 (divBase4 321 3) 21 = 2223 := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l1576_157640


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l1576_157609

theorem integer_pair_divisibility (x y : ℕ+) : 
  (((x : ℤ) * y - 6)^2 ∣ (x : ℤ)^2 + y^2) ↔ 
  ((x = 7 ∧ y = 1) ∨ (x = 4 ∧ y = 2) ∨ (x = 3 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l1576_157609


namespace NUMINAMATH_CALUDE_village_population_l1576_157692

/-- The population change over two years -/
def population_change (initial : ℝ) : ℝ := initial * 1.3 * 0.7

/-- The problem statement -/
theorem village_population : 
  ∃ (initial : ℝ), 
    population_change initial = 13650 ∧ 
    initial = 15000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1576_157692


namespace NUMINAMATH_CALUDE_ball_count_l1576_157681

theorem ball_count (red green blue total : ℕ) 
  (ratio : red = 15 ∧ green = 13 ∧ blue = 17)
  (red_count : red = 907) :
  total = 2721 :=
by sorry

end NUMINAMATH_CALUDE_ball_count_l1576_157681


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1576_157616

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given non-zero vectors a and b such that ‖a + 3b‖ = ‖a - 3b‖, 
    the angle between them is 90 degrees. -/
theorem angle_between_vectors (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : ‖a + 3 • b‖ = ‖a - 3 • b‖) : 
    Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1576_157616


namespace NUMINAMATH_CALUDE_cube_dimension_ratio_l1576_157628

theorem cube_dimension_ratio (v1 v2 : ℝ) (h1 : v1 = 27) (h2 : v2 = 216) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_dimension_ratio_l1576_157628


namespace NUMINAMATH_CALUDE_help_sign_white_area_l1576_157634

theorem help_sign_white_area :
  let sign_width : ℕ := 18
  let sign_height : ℕ := 7
  let h_area : ℕ := 13
  let e_area : ℕ := 11
  let l_area : ℕ := 8
  let p_area : ℕ := 11
  let total_black_area : ℕ := h_area + e_area + l_area + p_area
  let total_sign_area : ℕ := sign_width * sign_height
  total_sign_area - total_black_area = 83 := by
  sorry

end NUMINAMATH_CALUDE_help_sign_white_area_l1576_157634


namespace NUMINAMATH_CALUDE_stating_max_areas_theorem_l1576_157697

/-- Represents a circular disk divided by radii, a secant line, and a non-central chord -/
structure DividedDisk where
  n : ℕ
  radii_count : n > 0

/-- 
Calculates the maximum number of non-overlapping areas in a divided disk.
-/
def max_areas (disk : DividedDisk) : ℕ :=
  4 * disk.n + 1

/-- 
Theorem stating that the maximum number of non-overlapping areas in a divided disk
is equal to 4n + 1, where n is the number of equally spaced radii.
-/
theorem max_areas_theorem (disk : DividedDisk) :
  max_areas disk = 4 * disk.n + 1 := by sorry

end NUMINAMATH_CALUDE_stating_max_areas_theorem_l1576_157697


namespace NUMINAMATH_CALUDE_penguin_colony_fish_consumption_l1576_157667

theorem penguin_colony_fish_consumption (initial_size : ℕ) : 
  (2 * (2 * initial_size) + 129 = 1077) → 
  (initial_size = 158) := by
  sorry

end NUMINAMATH_CALUDE_penguin_colony_fish_consumption_l1576_157667


namespace NUMINAMATH_CALUDE_remainder_problem_l1576_157629

theorem remainder_problem (k : ℕ) (r : ℕ) (h1 : k > 0) (h2 : k < 38) 
  (h3 : k % 5 = 2) (h4 : k % 6 = 5) (h5 : k % 7 = r) (h6 : r < 7) : k % 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1576_157629


namespace NUMINAMATH_CALUDE_rex_cards_left_l1576_157644

-- Define the number of cards each person has
def nicole_cards : ℕ := 700
def cindy_cards : ℕ := (3 * nicole_cards + (40 * 3 * nicole_cards) / 100)
def tim_cards : ℕ := (4 * cindy_cards) / 5
def rex_joe_cards : ℕ := ((60 * (nicole_cards + cindy_cards + tim_cards)) / 100)

-- Define the number of people sharing Rex and Joe's cards
def num_sharing_people : ℕ := 9

-- Theorem to prove
theorem rex_cards_left : 
  (rex_joe_cards / num_sharing_people) = 399 := by sorry

end NUMINAMATH_CALUDE_rex_cards_left_l1576_157644


namespace NUMINAMATH_CALUDE_three_lines_cannot_form_triangle_l1576_157651

/-- Three lines in the plane -/
structure ThreeLines where
  l1 : ℝ → ℝ → Prop
  l2 : ℝ → ℝ → Prop
  l3 : ℝ → ℝ → ℝ → Prop

/-- The condition that three lines cannot form a triangle -/
def cannotFormTriangle (lines : ThreeLines) (m : ℝ) : Prop :=
  (∃ (x y : ℝ), lines.l1 x y ∧ lines.l2 x y ∧ lines.l3 m x y) ∨
  (∃ (a b : ℝ), ∀ (x y : ℝ), (lines.l1 x y ↔ y = a*x + b) ∧ 
                              (lines.l3 m x y ↔ y = a*x + (1 - a*m)/m)) ∨
  (∃ (a b : ℝ), ∀ (x y : ℝ), (lines.l2 x y ↔ y = a*x + b) ∧ 
                              (lines.l3 m x y ↔ y = a*x + (1 + a*m)/m))

/-- The given lines -/
def givenLines : ThreeLines :=
  { l1 := λ x y => 2*x - 3*y + 1 = 0
  , l2 := λ x y => 4*x + 3*y + 5 = 0
  , l3 := λ m x y => m*x - y - 1 = 0 }

theorem three_lines_cannot_form_triangle :
  {m : ℝ | cannotFormTriangle givenLines m} = {-4/3, 2/3, 4/3} := by sorry

end NUMINAMATH_CALUDE_three_lines_cannot_form_triangle_l1576_157651


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l1576_157676

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 7^n + 1 → 
  Odd n → 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) → 
  x = 16808 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l1576_157676


namespace NUMINAMATH_CALUDE_symmetrical_triangles_are_congruent_l1576_157617

/-- Two triangles are symmetrical about a line if each point of one triangle has a corresponding point in the other triangle that is equidistant from the line of symmetry. -/
def symmetrical_triangles (t1 t2 : Set Point) (l : Line) : Prop := sorry

/-- Two triangles are congruent if they have the same shape and size. -/
def congruent_triangles (t1 t2 : Set Point) : Prop := sorry

/-- If two triangles are symmetrical about a line, then they are congruent. -/
theorem symmetrical_triangles_are_congruent (t1 t2 : Set Point) (l : Line) :
  symmetrical_triangles t1 t2 l → congruent_triangles t1 t2 := by sorry

end NUMINAMATH_CALUDE_symmetrical_triangles_are_congruent_l1576_157617


namespace NUMINAMATH_CALUDE_ellipse_and_circle_tangent_lines_l1576_157688

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with center (x, y) and radius r -/
structure Circle where
  x : ℝ
  y : ℝ
  r : ℝ
  h_pos : 0 < r

theorem ellipse_and_circle_tangent_lines 
  (C : Ellipse) 
  (E : Circle)
  (h_minor : C.b^2 = 3)
  (h_focus : C.a^2 - C.b^2 = 3)
  (h_radius : E.r^2 = 2)
  (h_center : E.x^2 / C.a^2 + E.y^2 / C.b^2 = 1)
  (k₁ k₂ : ℝ)
  (h_tangent₁ : (E.x - x)^2 + (k₁ * x - E.y)^2 = E.r^2)
  (h_tangent₂ : (E.x - x)^2 + (k₂ * x - E.y)^2 = E.r^2) :
  (C.a^2 = 6 ∧ C.b^2 = 3) ∧ k₁ * k₂ = -1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_and_circle_tangent_lines_l1576_157688


namespace NUMINAMATH_CALUDE_given_curve_is_circle_l1576_157614

-- Define a polar coordinate
def PolarCoordinate := ℝ × ℝ

-- Define a circle in terms of its radius
def Circle (radius : ℝ) := {p : PolarCoordinate | p.2 = radius}

-- Define the curve given by the equation r = 5
def GivenCurve := {p : PolarCoordinate | p.2 = 5}

-- Theorem statement
theorem given_curve_is_circle : GivenCurve = Circle 5 := by
  sorry

end NUMINAMATH_CALUDE_given_curve_is_circle_l1576_157614


namespace NUMINAMATH_CALUDE_circles_intersect_l1576_157695

/-- Two circles are intersecting if the distance between their centers is less than the sum of their radii
    and greater than the absolute difference of their radii. -/
def are_intersecting (r₁ r₂ d : ℝ) : Prop :=
  d < r₁ + r₂ ∧ d > |r₁ - r₂|

/-- Given two circles with radii 3 and 5, whose centers are 2 units apart, prove they are intersecting. -/
theorem circles_intersect : are_intersecting 3 5 2 := by
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_circles_intersect_l1576_157695


namespace NUMINAMATH_CALUDE_radical_subtraction_l1576_157624

theorem radical_subtraction : (5 / Real.sqrt 2) - Real.sqrt (1 / 2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_radical_subtraction_l1576_157624


namespace NUMINAMATH_CALUDE_cookies_for_students_minimum_recipes_needed_l1576_157670

/-- Calculates the minimum number of full recipes needed to provide cookies for students -/
theorem cookies_for_students (original_students : ℕ) (increase_percent : ℕ) 
  (cookies_per_student : ℕ) (cookies_per_recipe : ℕ) : ℕ :=
  let new_students := original_students * (100 + increase_percent) / 100
  let total_cookies := new_students * cookies_per_student
  let recipes_needed := (total_cookies + cookies_per_recipe - 1) / cookies_per_recipe
  recipes_needed

/-- The minimum number of full recipes needed for the given conditions is 33 -/
theorem minimum_recipes_needed : 
  cookies_for_students 108 50 3 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_cookies_for_students_minimum_recipes_needed_l1576_157670


namespace NUMINAMATH_CALUDE_twenty_team_tournament_matches_l1576_157613

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  num_matches : ℕ

/-- Calculates the number of matches needed in a single-elimination tournament. -/
def matches_needed (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 20 teams requires 19 matches. -/
theorem twenty_team_tournament_matches :
  ∀ t : Tournament, t.num_teams = 20 → t.num_matches = matches_needed t.num_teams := by
  sorry

end NUMINAMATH_CALUDE_twenty_team_tournament_matches_l1576_157613


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l1576_157674

theorem x_positive_sufficient_not_necessary_for_x_nonzero :
  (∃ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l1576_157674


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l1576_157679

/-- A cylinder with a rectangular front view of area 6 has a lateral area of 6π -/
theorem cylinder_lateral_area (h : ℝ) (h_pos : h > 0) : 
  let d := 6 / h  -- diameter of the base
  let lateral_area := π * d * h
  lateral_area = 6 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l1576_157679


namespace NUMINAMATH_CALUDE_number_equality_l1576_157647

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (49/216) * (1/x)) : x = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1576_157647


namespace NUMINAMATH_CALUDE_matrix_operation_proof_l1576_157690

theorem matrix_operation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -3; 0, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-7, 9; 6, -10]
  2 • A + B = !![1, 3; 6, 0] := by
  sorry

end NUMINAMATH_CALUDE_matrix_operation_proof_l1576_157690


namespace NUMINAMATH_CALUDE_rook_tour_existence_l1576_157606

/-- A rook move on an m × n board. -/
inductive RookMove
  | up : RookMove
  | right : RookMove
  | down : RookMove
  | left : RookMove

/-- A valid sequence of rook moves on an m × n board. -/
def ValidMoveSequence (m n : ℕ) : List RookMove → Prop :=
  sorry

/-- A sequence of moves visits all squares exactly once and returns to start. -/
def VisitsAllSquaresOnce (m n : ℕ) (moves : List RookMove) : Prop :=
  sorry

theorem rook_tour_existence (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (∃ moves : List RookMove, ValidMoveSequence m n moves ∧ VisitsAllSquaresOnce m n moves) ↔
  (Even m ∧ Even n) :=
sorry

end NUMINAMATH_CALUDE_rook_tour_existence_l1576_157606


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_fifteen_fourths_l1576_157627

theorem floor_plus_self_eq_fifteen_fourths :
  ∃! (x : ℚ), (⌊x⌋ : ℚ) + x = 15/4 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_fifteen_fourths_l1576_157627


namespace NUMINAMATH_CALUDE_circle_a_range_l1576_157638

/-- A circle in the xy-plane is represented by the equation (x^2 + y^2 + 2x - 4y + a = 0) -/
def is_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + a = 0

/-- The range of a for which the equation represents a circle -/
theorem circle_a_range :
  {a : ℝ | is_circle a} = Set.Iio 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_a_range_l1576_157638


namespace NUMINAMATH_CALUDE_alice_has_largest_result_l1576_157663

def initial_number : ℕ := 15

def alice_result (n : ℕ) : ℕ := n * 3 - 2 + 4

def bob_result (n : ℕ) : ℕ := n * 2 + 3 - 5

def charlie_result (n : ℕ) : ℕ := ((n + 5) / 2) * 4

theorem alice_has_largest_result :
  alice_result initial_number > bob_result initial_number ∧
  alice_result initial_number > charlie_result initial_number := by
  sorry

end NUMINAMATH_CALUDE_alice_has_largest_result_l1576_157663


namespace NUMINAMATH_CALUDE_bills_profit_percentage_l1576_157656

/-- Represents the original profit percentage -/
def original_profit_percentage : ℝ := 10

/-- Represents the original selling price -/
def original_selling_price : ℝ := 549.9999999999995

/-- Represents the additional profit if the product was bought for 10% less and sold at 30% profit -/
def additional_profit : ℝ := 35

theorem bills_profit_percentage :
  let P := original_selling_price / (1 + original_profit_percentage / 100)
  let new_selling_price := P * 0.9 * 1.3
  new_selling_price = original_selling_price + additional_profit :=
by sorry

end NUMINAMATH_CALUDE_bills_profit_percentage_l1576_157656


namespace NUMINAMATH_CALUDE_range_of_a_l1576_157615

open Set

/-- The equation that must have 3 distinct real solutions -/
def equation (a x : ℝ) : ℝ := 2 * x * |x| - (a - 2) * x + |x| - a + 1

/-- The condition that the equation has 3 distinct real solutions -/
def has_three_distinct_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    equation a x₁ = 0 ∧ equation a x₂ = 0 ∧ equation a x₃ = 0

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  has_three_distinct_solutions a → a ∈ Ioi 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1576_157615


namespace NUMINAMATH_CALUDE_thief_speed_calculation_chase_problem_l1576_157684

/-- Represents the chase scenario between a policeman and a thief -/
structure ChaseScenario where
  initial_distance : ℝ  -- in meters
  policeman_speed : ℝ   -- in km/hr
  thief_distance : ℝ    -- in meters
  thief_speed : ℝ       -- in km/hr

/-- Theorem stating the relationship between the given parameters and the thief's speed -/
theorem thief_speed_calculation (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 160)
  (h2 : scenario.policeman_speed = 10)
  (h3 : scenario.thief_distance = 640) :
  scenario.thief_speed = 8 := by
  sorry

/-- Main theorem proving the specific case -/
theorem chase_problem : 
  ∃ (scenario : ChaseScenario), 
    scenario.initial_distance = 160 ∧ 
    scenario.policeman_speed = 10 ∧ 
    scenario.thief_distance = 640 ∧ 
    scenario.thief_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_thief_speed_calculation_chase_problem_l1576_157684


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1576_157603

theorem shaded_area_calculation (large_square_area medium_square_area small_square_area : ℝ)
  (h1 : large_square_area = 49)
  (h2 : medium_square_area = 25)
  (h3 : small_square_area = 9) :
  small_square_area + (large_square_area - medium_square_area) = 33 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1576_157603


namespace NUMINAMATH_CALUDE_certain_number_equation_l1576_157602

theorem certain_number_equation (x : ℝ) : ((x + 2 - 6) * 3) / 4 = 3 ↔ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1576_157602


namespace NUMINAMATH_CALUDE_race_distance_l1576_157694

/-- Given two runners in a race, prove the total distance of the race -/
theorem race_distance (time_A time_B : ℝ) (lead : ℝ) (h1 : time_A = 30) (h2 : time_B = 45) (h3 : lead = 33.333333333333336) :
  ∃ (distance : ℝ), distance = 100 ∧ distance / time_A - distance / time_B = lead / time_A := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1576_157694


namespace NUMINAMATH_CALUDE_ten_cubes_shaded_l1576_157671

/-- Represents a 4x4x4 cube with a specific shading pattern -/
structure ShadedCube where
  /-- Total number of smaller cubes -/
  total_cubes : Nat
  /-- Number of cubes per edge -/
  edge_length : Nat
  /-- Number of shaded cubes per face -/
  shaded_per_face : Nat
  /-- Condition: total cubes is 64 -/
  total_is_64 : total_cubes = 64
  /-- Condition: edge length is 4 -/
  edge_is_4 : edge_length = 4
  /-- Condition: 5 cubes are shaded per face -/
  five_shaded : shaded_per_face = 5

/-- The number of uniquely shaded cubes in the ShadedCube -/
def uniquely_shaded_cubes (c : ShadedCube) : Nat :=
  8 + 2  -- 8 corner cubes + 2 center cubes on opposite faces

/-- Theorem stating that exactly 10 cubes are uniquely shaded -/
theorem ten_cubes_shaded (c : ShadedCube) :
  uniquely_shaded_cubes c = 10 := by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ten_cubes_shaded_l1576_157671


namespace NUMINAMATH_CALUDE_product_equals_720_l1576_157610

theorem product_equals_720 (n : ℕ) (h : n = 5) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_720_l1576_157610


namespace NUMINAMATH_CALUDE_range_of_m_l1576_157621

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the property of having two roots in [0, 2]
def has_two_roots_in_interval (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ 2 ∧ 
  f x₁ + m = 0 ∧ f x₂ + m = 0

-- Theorem statement
theorem range_of_m (m : ℝ) :
  has_two_roots_in_interval m → 0 ≤ m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1576_157621


namespace NUMINAMATH_CALUDE_equation_solution_l1576_157639

theorem equation_solution (x : ℝ) : (x + 2)^(x + 3) = 1 → x = -3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1576_157639


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1576_157650

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = Complex.abs (1 - Complex.I)) :
  z.im = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1576_157650


namespace NUMINAMATH_CALUDE_dividend_divisor_problem_l1576_157683

theorem dividend_divisor_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  y / x = 6 ∧ x + y + 6 = 216 → x = 30 ∧ y = 180 := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisor_problem_l1576_157683


namespace NUMINAMATH_CALUDE_river_width_l1576_157641

/-- Given a river with specified depth, flow rate, and volume per minute, prove its width. -/
theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) :
  depth = 2 →
  flow_rate = 3 →
  volume_per_minute = 4500 →
  (flow_rate * 1000 / 60) * depth * (volume_per_minute / (flow_rate * 1000 / 60) / depth) = 45 := by
  sorry

end NUMINAMATH_CALUDE_river_width_l1576_157641


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_17_l1576_157677

theorem modular_inverse_of_5_mod_17 : 
  ∃! x : ℕ, x ∈ Finset.range 17 ∧ (5 * x) % 17 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_17_l1576_157677


namespace NUMINAMATH_CALUDE_square_cube_remainder_l1576_157620

theorem square_cube_remainder (a n : ℕ) 
  (h1 : a^2 % n = 8)
  (h2 : a^3 % n = 25)
  (h3 : n > 25) :
  n = 113 := by
sorry

end NUMINAMATH_CALUDE_square_cube_remainder_l1576_157620


namespace NUMINAMATH_CALUDE_total_dogs_count_l1576_157653

/-- Represents the number of dogs in the Smartpup Training Center -/
structure DogCount where
  sit : ℕ
  stay : ℕ
  roll_over : ℕ
  jump : ℕ
  sit_stay : ℕ
  stay_roll : ℕ
  sit_roll : ℕ
  jump_stay : ℕ
  sit_stay_roll : ℕ
  no_tricks : ℕ

/-- Theorem stating that the total number of dogs is 150 given the specified conditions -/
theorem total_dogs_count (d : DogCount) 
  (h1 : d.sit = 60)
  (h2 : d.stay = 40)
  (h3 : d.roll_over = 45)
  (h4 : d.jump = 50)
  (h5 : d.sit_stay = 25)
  (h6 : d.stay_roll = 15)
  (h7 : d.sit_roll = 20)
  (h8 : d.jump_stay = 5)
  (h9 : d.sit_stay_roll = 10)
  (h10 : d.no_tricks = 5) : 
  d.sit + d.stay + d.roll_over + d.jump - d.sit_stay - d.stay_roll - d.sit_roll - 
  d.jump_stay + d.sit_stay_roll + d.no_tricks = 150 := by
  sorry


end NUMINAMATH_CALUDE_total_dogs_count_l1576_157653


namespace NUMINAMATH_CALUDE_parabola_c_value_l1576_157608

/-- A parabola with equation x = ay^2 + by + c, vertex at (4, 1), and passing through (1, 3) -/
def Parabola (a b c : ℝ) : Prop :=
  ∀ y : ℝ, 4 = a * 1^2 + b * 1 + c ∧
            1 = a * 3^2 + b * 3 + c

theorem parabola_c_value :
  ∀ a b c : ℝ, Parabola a b c → c = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1576_157608


namespace NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l1576_157625

theorem quadratic_discriminant_nonnegative (a b : ℝ) :
  (∃ x : ℝ, x^2 + a*x + b ≤ 0) → a^2 - 4*b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l1576_157625


namespace NUMINAMATH_CALUDE_absolute_value_equation_sum_l1576_157605

theorem absolute_value_equation_sum (n : ℝ) : 
  (∃ n₁ n₂ : ℝ, |3 * n₁ - 8| = 5 ∧ |3 * n₂ - 8| = 5 ∧ n₁ ≠ n₂ ∧ n₁ + n₂ = 16/3) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_sum_l1576_157605


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1576_157652

theorem fraction_subtraction : 
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1576_157652


namespace NUMINAMATH_CALUDE_f_equals_g_l1576_157664

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (m : ℝ) : ℝ := m^2 - 2*m - 1

theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l1576_157664


namespace NUMINAMATH_CALUDE_tower_block_count_l1576_157637

/-- The total number of blocks in a tower after adding more blocks -/
def total_blocks (initial : Float) (added : Float) : Float :=
  initial + added

/-- Theorem: The total number of blocks is the sum of initial and added blocks -/
theorem tower_block_count (initial : Float) (added : Float) :
  total_blocks initial added = initial + added := by
  sorry

end NUMINAMATH_CALUDE_tower_block_count_l1576_157637


namespace NUMINAMATH_CALUDE_polyhedron_property_l1576_157661

/-- A convex polyhedron with specific face and vertex properties -/
structure ConvexPolyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  hexagons : ℕ
  vertices : ℕ
  P : ℕ  -- number of pentagons meeting at each vertex
  H : ℕ  -- number of hexagons meeting at each vertex
  T : ℕ  -- number of triangles meeting at each vertex

/-- The properties of the specific polyhedron in the problem -/
def problem_polyhedron : ConvexPolyhedron where
  faces := 38
  triangles := 20
  pentagons := 10
  hexagons := 8
  vertices := 115
  P := 4
  H := 2
  T := 2

/-- The theorem to be proved -/
theorem polyhedron_property (poly : ConvexPolyhedron) 
  (h1 : poly.faces = 38)
  (h2 : poly.triangles = 2 * poly.pentagons)
  (h3 : poly.hexagons = 8)
  (h4 : poly.P = 2 * poly.H)
  (h5 : poly.faces = poly.triangles + poly.pentagons + poly.hexagons)
  (h6 : poly = problem_polyhedron) :
  100 * poly.P + 10 * poly.T + poly.vertices = 535 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l1576_157661


namespace NUMINAMATH_CALUDE_adult_ticket_price_is_60_l1576_157630

/-- Represents the ticket prices and attendance for a football game -/
structure FootballGame where
  adultTicketPrice : ℕ
  childTicketPrice : ℕ
  totalAttendance : ℕ
  adultAttendance : ℕ
  totalRevenue : ℕ

/-- Theorem stating that the adult ticket price is 60 cents -/
theorem adult_ticket_price_is_60 (game : FootballGame) :
  game.childTicketPrice = 25 ∧
  game.totalAttendance = 280 ∧
  game.totalRevenue = 14000 ∧
  game.adultAttendance = 200 →
  game.adultTicketPrice = 60 := by
  sorry

#check adult_ticket_price_is_60

end NUMINAMATH_CALUDE_adult_ticket_price_is_60_l1576_157630


namespace NUMINAMATH_CALUDE_mk_97_check_one_l1576_157600

theorem mk_97_check_one (x : ℝ) : x = 1 ↔ x ≠ 2 * x ∧ ∃! y : ℝ, y ^ 2 + 2 * x * y + x = 0 := by sorry

end NUMINAMATH_CALUDE_mk_97_check_one_l1576_157600


namespace NUMINAMATH_CALUDE_tricycle_count_proof_l1576_157623

/-- Represents the number of wheels on a vehicle -/
def wheels : Nat → Nat
  | 0 => 2  -- bicycle
  | 1 => 3  -- tricycle
  | 2 => 2  -- scooter
  | _ => 0  -- undefined for other values

/-- Represents the count of each type of vehicle -/
structure VehicleCounts where
  bicycles : Nat
  tricycles : Nat
  scooters : Nat

theorem tricycle_count_proof (counts : VehicleCounts) : 
  counts.bicycles + counts.tricycles + counts.scooters = 10 →
  wheels 0 * counts.bicycles + wheels 1 * counts.tricycles + wheels 2 * counts.scooters = 29 →
  counts.tricycles = 9 := by
  sorry

#check tricycle_count_proof

end NUMINAMATH_CALUDE_tricycle_count_proof_l1576_157623


namespace NUMINAMATH_CALUDE_quarter_circles_sum_limit_l1576_157698

theorem quarter_circles_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |n * (π * D / (4 * n)) - (π * D / 4)| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_limit_l1576_157698


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l1576_157635

theorem binomial_expansion_constant_term (a b : ℝ) (n : ℕ) :
  (2 : ℝ) ^ n = 4 →
  n = 2 →
  (a + b) ^ n = a ^ 2 + 2 * a * b + 9 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l1576_157635


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_14_l1576_157686

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the final parabola after transformations
def final_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

-- Define the zeros of the final parabola
def p : ℝ := 8
def q : ℝ := 6

-- Theorem statement
theorem sum_of_zeros_is_14 : p + q = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_14_l1576_157686


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1576_157685

def p (x : ℝ) : Prop := x = 1

def q (x : ℝ) : Prop := x^3 - 2*x + 1 = 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1576_157685


namespace NUMINAMATH_CALUDE_kylie_coins_problem_l1576_157626

/-- The number of coins Kylie's father gave her -/
def coins_from_father : ℕ := sorry

theorem kylie_coins_problem : coins_from_father = 8 := by
  have piggy_bank : ℕ := 15
  have from_brother : ℕ := 13
  have given_to_laura : ℕ := 21
  have left_with : ℕ := 15
  
  have total_before_father : ℕ := piggy_bank + from_brother
  have total_after_father : ℕ := total_before_father + coins_from_father
  have after_giving_to_laura : ℕ := total_after_father - given_to_laura
  
  have : after_giving_to_laura = left_with := by sorry
  
  sorry

end NUMINAMATH_CALUDE_kylie_coins_problem_l1576_157626


namespace NUMINAMATH_CALUDE_regression_line_equation_l1576_157632

/-- Given a regression line with slope -1 passing through the point (1, 2),
    prove that its equation is y = -x + 3 -/
theorem regression_line_equation (slope : ℝ) (center : ℝ × ℝ) :
  slope = -1 →
  center = (1, 2) →
  ∀ x y : ℝ, y = slope * (x - center.1) + center.2 ↔ y = -x + 3 :=
by sorry

end NUMINAMATH_CALUDE_regression_line_equation_l1576_157632


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l1576_157666

/-- Represents a square quilt made of smaller squares -/
structure Quilt :=
  (size : Nat)
  (shaded_row : Nat)
  (shaded_column : Nat)

/-- Calculates the fraction of shaded area in a quilt -/
def shaded_fraction (q : Quilt) : Rat :=
  let total_squares := q.size * q.size
  let shaded_squares := q.size + q.size - 1
  shaded_squares / total_squares

/-- Theorem stating that for a 4x4 quilt with one shaded row and column, 
    the shaded fraction is 7/16 -/
theorem quilt_shaded_fraction :
  ∀ (q : Quilt), q.size = 4 → shaded_fraction q = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l1576_157666


namespace NUMINAMATH_CALUDE_complex_number_equivalence_l1576_157672

theorem complex_number_equivalence : 
  let z : ℂ := (1 - I) / (2 + I)
  z = 1/5 - 3/5*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_equivalence_l1576_157672


namespace NUMINAMATH_CALUDE_car_travel_time_l1576_157659

/-- Proves that a car traveling at 160 km/h for 800 km takes 5 hours -/
theorem car_travel_time (speed : ℝ) (distance : ℝ) (h1 : speed = 160) (h2 : distance = 800) :
  distance / speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_l1576_157659
