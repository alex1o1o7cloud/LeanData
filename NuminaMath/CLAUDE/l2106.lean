import Mathlib

namespace NUMINAMATH_CALUDE_reggie_has_70_marbles_l2106_210644

/-- Calculates the number of marbles Reggie has after playing a series of games -/
def reggies_marbles (total_games : ℕ) (lost_games : ℕ) (marbles_per_game : ℕ) : ℕ :=
  (total_games - lost_games) * marbles_per_game - lost_games * marbles_per_game

/-- Proves that Reggie has 70 marbles after playing 9 games, losing 1, with 10 marbles bet per game -/
theorem reggie_has_70_marbles :
  reggies_marbles 9 1 10 = 70 := by
  sorry

#eval reggies_marbles 9 1 10

end NUMINAMATH_CALUDE_reggie_has_70_marbles_l2106_210644


namespace NUMINAMATH_CALUDE_decimal_representation_5_11_l2106_210603

/-- The decimal representation of 5/11 has a repeating sequence of length 2 -/
def repeating_length : ℕ := 2

/-- The 150th decimal place in the representation of 5/11 -/
def decimal_place : ℕ := 150

/-- The result we want to prove -/
def result : ℕ := 5

theorem decimal_representation_5_11 :
  (decimal_place % repeating_length = 0) ∧
  (result = 5) := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_5_11_l2106_210603


namespace NUMINAMATH_CALUDE_games_for_512_participants_l2106_210685

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  participants : ℕ
  is_power_of_two : ∃ n : ℕ, participants = 2^n

/-- Calculates the number of games required to determine a winner in a single-elimination tournament. -/
def games_required (tournament : SingleEliminationTournament) : ℕ :=
  tournament.participants - 1

/-- Theorem stating that a single-elimination tournament with 512 participants requires 511 games. -/
theorem games_for_512_participants :
  ∀ (tournament : SingleEliminationTournament),
  tournament.participants = 512 →
  games_required tournament = 511 := by
  sorry

#eval games_required ⟨512, ⟨9, rfl⟩⟩

end NUMINAMATH_CALUDE_games_for_512_participants_l2106_210685


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l2106_210632

theorem solution_implies_m_value (x m : ℝ) :
  x = 1 → 2 * x + m - 6 = 0 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l2106_210632


namespace NUMINAMATH_CALUDE_wheel_revolutions_for_one_mile_l2106_210661

-- Define the wheel's diameter
def wheel_diameter : ℝ := 8

-- Define the length of a mile in feet
def mile_in_feet : ℝ := 5280

-- Theorem statement
theorem wheel_revolutions_for_one_mile :
  let wheel_circumference := π * wheel_diameter
  let revolutions := mile_in_feet / wheel_circumference
  revolutions = 660 / π :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_revolutions_for_one_mile_l2106_210661


namespace NUMINAMATH_CALUDE_expected_vote_percentage_is_47_percent_l2106_210663

/-- The percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.60

/-- The percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 1 - democrat_percentage

/-- The percentage of registered Democrat voters expected to vote for candidate A -/
def democrat_vote_percentage : ℝ := 0.65

/-- The percentage of registered Republican voters expected to vote for candidate A -/
def republican_vote_percentage : ℝ := 0.20

/-- The expected percentage of registered voters who will vote for candidate A -/
def expected_vote_percentage : ℝ :=
  democrat_percentage * democrat_vote_percentage +
  republican_percentage * republican_vote_percentage

theorem expected_vote_percentage_is_47_percent :
  expected_vote_percentage = 0.47 :=
sorry

end NUMINAMATH_CALUDE_expected_vote_percentage_is_47_percent_l2106_210663


namespace NUMINAMATH_CALUDE_total_savings_l2106_210638

/-- The total savings over two months given the savings in September and the difference in October -/
theorem total_savings (september : ℕ) (difference : ℕ) : 
  september = 260 → difference = 30 → september + (september + difference) = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_l2106_210638


namespace NUMINAMATH_CALUDE_apple_pies_theorem_l2106_210601

def total_apples : ℕ := 128
def unripe_apples : ℕ := 23
def apples_per_pie : ℕ := 7

theorem apple_pies_theorem : 
  (total_apples - unripe_apples) / apples_per_pie = 15 :=
by sorry

end NUMINAMATH_CALUDE_apple_pies_theorem_l2106_210601


namespace NUMINAMATH_CALUDE_sin_15_times_sin_75_l2106_210654

theorem sin_15_times_sin_75 : Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_times_sin_75_l2106_210654


namespace NUMINAMATH_CALUDE_count_distinguishable_triangles_l2106_210641

/-- Represents the number of available colors for small triangles -/
def num_colors : ℕ := 8

/-- Represents a large equilateral triangle constructed from four smaller triangles -/
structure LargeTriangle where
  corner1 : Fin num_colors
  corner2 : Fin num_colors
  corner3 : Fin num_colors
  center : Fin num_colors

/-- Two large triangles are considered equivalent if they can be matched by rotations or reflections -/
def equivalent (t1 t2 : LargeTriangle) : Prop :=
  ∃ (perm : Fin 3 → Fin 3), 
    (t1.corner1 = t2.corner1 ∧ t1.corner2 = t2.corner2 ∧ t1.corner3 = t2.corner3) ∨
    (t1.corner1 = t2.corner2 ∧ t1.corner2 = t2.corner3 ∧ t1.corner3 = t2.corner1) ∨
    (t1.corner1 = t2.corner3 ∧ t1.corner2 = t2.corner1 ∧ t1.corner3 = t2.corner2)

/-- The set of all distinguishable large triangles -/
def distinguishable_triangles : Finset LargeTriangle :=
  sorry

theorem count_distinguishable_triangles : 
  Finset.card distinguishable_triangles = 960 := by
  sorry

end NUMINAMATH_CALUDE_count_distinguishable_triangles_l2106_210641


namespace NUMINAMATH_CALUDE_complement_implies_a_value_l2106_210643

def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}
def P (a : ℝ) : Set ℝ := {2, a^2 + 2 - a}

theorem complement_implies_a_value (a : ℝ) : 
  (U a \ P a = {-1}) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_complement_implies_a_value_l2106_210643


namespace NUMINAMATH_CALUDE_sin_plus_cos_for_point_l2106_210605

/-- Given that the terminal side of angle θ passes through point P(-3,4),
    prove that sin θ + cos θ = 1/5 -/
theorem sin_plus_cos_for_point (θ : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = -3 ∧ r * Real.sin θ = 4) →
  Real.sin θ + Real.cos θ = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_for_point_l2106_210605


namespace NUMINAMATH_CALUDE_building_heights_sum_l2106_210666

/-- Proves that the total height of three buildings is 340 feet -/
theorem building_heights_sum (middle_height : ℝ) (left_height : ℝ) (right_height : ℝ)
  (h1 : middle_height = 100)
  (h2 : left_height = 0.8 * middle_height)
  (h3 : right_height = left_height + middle_height - 20) :
  left_height + middle_height + right_height = 340 := by
  sorry

end NUMINAMATH_CALUDE_building_heights_sum_l2106_210666


namespace NUMINAMATH_CALUDE_total_cloud_count_l2106_210618

def cloud_count (carson_funny : ℕ) (brother_multiplier : ℕ) (sister_divisor : ℕ) (cousin_multiplier : ℕ) : ℕ :=
  let brother_dragons := carson_funny * brother_multiplier
  let sister_sailboats := carson_funny / sister_divisor
  let cousin_birds := cousin_multiplier * (carson_funny + sister_sailboats)
  carson_funny + brother_dragons + sister_sailboats + cousin_birds

theorem total_cloud_count :
  cloud_count 12 5 2 2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_cloud_count_l2106_210618


namespace NUMINAMATH_CALUDE_function_growth_l2106_210659

/-- For any differentiable function f: ℝ → ℝ, if f'(x) > f(x) for all x ∈ ℝ,
    then f(a) > e^a * f(0) for any a > 0. -/
theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) :
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l2106_210659


namespace NUMINAMATH_CALUDE_min_value_expression_l2106_210658

theorem min_value_expression (x y z t : ℝ) 
  (h1 : x + 4*y = 4) 
  (h2 : y > 0) 
  (h3 : 0 < t) 
  (h4 : t < z) : 
  (4*z^2 / abs x) + (abs (x*z^2) / y) + (12 / (t*(z-t))) ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2106_210658


namespace NUMINAMATH_CALUDE_triangle_problem_l2106_210604

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  Real.cos B = 7/9 →
  a * c * Real.cos B = 7 →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  b = 2 ∧ Real.sin (A - B) = 10 * Real.sqrt 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2106_210604


namespace NUMINAMATH_CALUDE_system_of_equations_range_l2106_210679

theorem system_of_equations_range (x y k : ℝ) : 
  x - y = k - 1 →
  3 * x + 2 * y = 4 * k + 5 →
  2 * x + 3 * y > 7 →
  k > 1/3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_range_l2106_210679


namespace NUMINAMATH_CALUDE_curve_family_condition_l2106_210697

/-- A family of curves parameterized by p -/
def curve_family (p x y : ℝ) : Prop :=
  y = p^2 + (2*p - 1)*x + 2*x^2

/-- The condition for a point (x, y) to have at least one curve passing through it -/
def has_curve_passing_through (x y : ℝ) : Prop :=
  ∃ p : ℝ, curve_family p x y

/-- The theorem stating the equivalence between the existence of a curve passing through (x, y) 
    and the inequality y ≥ x² - x -/
theorem curve_family_condition (x y : ℝ) : 
  has_curve_passing_through x y ↔ y ≥ x^2 - x :=
sorry

end NUMINAMATH_CALUDE_curve_family_condition_l2106_210697


namespace NUMINAMATH_CALUDE_pencil_difference_l2106_210615

/-- The number of pencils Paige has in her desk -/
def pencils_in_desk : ℕ := 2

/-- The number of pencils Paige has in her backpack -/
def pencils_in_backpack : ℕ := 2

/-- The number of pencils Paige has at home -/
def pencils_at_home : ℕ := 15

/-- The difference between the number of pencils at Paige's home and in Paige's backpack -/
theorem pencil_difference : pencils_at_home - pencils_in_backpack = 13 := by
  sorry

end NUMINAMATH_CALUDE_pencil_difference_l2106_210615


namespace NUMINAMATH_CALUDE_symmetrical_shape_three_equal_parts_l2106_210616

/-- A symmetrical 2D shape -/
structure SymmetricalShape where
  area : ℝ
  height : ℝ
  width : ℝ
  is_symmetrical : Bool

/-- A straight cut on the shape -/
inductive Cut
  | Vertical : ℝ → Cut  -- position along width
  | Horizontal : ℝ → Cut  -- position along height

/-- Result of applying cuts to a shape -/
def apply_cuts (shape : SymmetricalShape) (cuts : List Cut) : List ℝ :=
  sorry

theorem symmetrical_shape_three_equal_parts (shape : SymmetricalShape) :
  shape.is_symmetrical →
  ∃ (vertical_cut : Cut) (horizontal_cut : Cut),
    vertical_cut = Cut.Vertical (shape.width / 2) ∧
    horizontal_cut = Cut.Horizontal (shape.height / 3) ∧
    apply_cuts shape [vertical_cut, horizontal_cut] = [shape.area / 3, shape.area / 3, shape.area / 3] :=
  sorry

end NUMINAMATH_CALUDE_symmetrical_shape_three_equal_parts_l2106_210616


namespace NUMINAMATH_CALUDE_halloween_cleaning_time_l2106_210698

/-- Calculates the total cleaning time for Halloween pranks -/
theorem halloween_cleaning_time 
  (egg_cleaning_time : ℕ) 
  (tp_cleaning_time : ℕ) 
  (num_eggs : ℕ) 
  (num_tp : ℕ) : 
  egg_cleaning_time = 15 ∧ 
  tp_cleaning_time = 30 ∧ 
  num_eggs = 60 ∧ 
  num_tp = 7 → 
  (num_eggs * egg_cleaning_time) / 60 + num_tp * tp_cleaning_time = 225 := by
  sorry

#check halloween_cleaning_time

end NUMINAMATH_CALUDE_halloween_cleaning_time_l2106_210698


namespace NUMINAMATH_CALUDE_joes_dad_marshmallow_fraction_l2106_210680

theorem joes_dad_marshmallow_fraction :
  ∀ (dad_marshmallows joe_marshmallows dad_roasted joe_roasted total_roasted : ℕ),
    dad_marshmallows = 21 →
    joe_marshmallows = 4 * dad_marshmallows →
    joe_roasted = joe_marshmallows / 2 →
    total_roasted = 49 →
    total_roasted = joe_roasted + dad_roasted →
    (dad_roasted : ℚ) / dad_marshmallows = 1 / 3 := by
  sorry

#check joes_dad_marshmallow_fraction

end NUMINAMATH_CALUDE_joes_dad_marshmallow_fraction_l2106_210680


namespace NUMINAMATH_CALUDE_complex_square_magnitude_l2106_210684

theorem complex_square_magnitude (z : ℂ) (h : z^2 + Complex.abs z^2 = 8 - 3*I) : 
  Complex.abs z^2 = 73/16 := by sorry

end NUMINAMATH_CALUDE_complex_square_magnitude_l2106_210684


namespace NUMINAMATH_CALUDE_square_sum_representation_l2106_210693

theorem square_sum_representation (x y : ℕ) (h : x ≠ y) :
  ∃ u v : ℕ, x^2 + x*y + y^2 = u^2 + 3*v^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_representation_l2106_210693


namespace NUMINAMATH_CALUDE_five_lines_sixteen_sections_l2106_210600

/-- The number of sections created by drawing n line segments through a rectangle,
    assuming each new line intersects all previous lines. -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else max_sections (n - 1) + n

/-- The theorem stating that 5 line segments create 16 sections in a rectangle. -/
theorem five_lines_sixteen_sections :
  max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_lines_sixteen_sections_l2106_210600


namespace NUMINAMATH_CALUDE_dog_walking_distance_l2106_210608

theorem dog_walking_distance (total_weekly_miles : ℝ) (dog1_daily_miles : ℝ) (dog2_daily_miles : ℝ) :
  total_weekly_miles = 70 ∧ dog1_daily_miles = 2 →
  dog2_daily_miles = 8 := by
sorry

end NUMINAMATH_CALUDE_dog_walking_distance_l2106_210608


namespace NUMINAMATH_CALUDE_beth_candies_theorem_l2106_210675

def total_candies : ℕ := 10

def is_valid_distribution (a b c : ℕ) : Prop :=
  a + b + c = total_candies ∧ a ≥ 3 ∧ b ≥ 2 ∧ c ≥ 2 ∧ c ≤ 3

def possible_beth_candies : Set ℕ := {2, 3, 4, 5}

theorem beth_candies_theorem :
  ∀ b : ℕ, (∃ a c : ℕ, is_valid_distribution a b c) ↔ b ∈ possible_beth_candies :=
sorry

end NUMINAMATH_CALUDE_beth_candies_theorem_l2106_210675


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_and_fixed_point_l2106_210673

noncomputable section

-- Define the ellipse Γ
def Γ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

-- Define the circle E
def E (x y : ℝ) : Prop := x^2 + (y - 3/2)^2 = 4

-- Define point D
def D : ℝ × ℝ := (0, -1/2)

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define symmetry about y-axis
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem ellipse_eccentricity_and_fixed_point 
  (a : ℝ) (A B : ℝ × ℝ) 
  (h1 : a > 1)
  (h2 : Γ a A.1 A.2)
  (h3 : Γ a B.1 B.2)
  (h4 : E A.1 A.2)
  (h5 : E B.1 B.2)
  (h6 : distance A B = 2 * Real.sqrt 3) :
  (∃ (e : ℝ), e = Real.sqrt 3 / 2 ∧ 
    e^2 = 1 - 1/a^2) ∧ 
  (∀ (M N N' : ℝ × ℝ), 
    Γ a M.1 M.2 → Γ a N.1 N.2 → symmetric_about_y_axis N N' →
    (∃ (k : ℝ), M.2 - D.2 = k * (M.1 - D.1) ∧ 
                N.2 - D.2 = k * (N.1 - D.1)) →
    ∃ (t : ℝ), M.2 - N'.2 = (M.1 - N'.1) * (0 - M.1) / (t - M.1) ∧ 
               t = 0 ∧ M.2 - (0 - M.1) * (M.2 - N'.2) / (M.1 - N'.1) = -2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_and_fixed_point_l2106_210673


namespace NUMINAMATH_CALUDE_truck_speed_truck_speed_is_52_l2106_210653

/-- The speed of the truck given two cars with different speeds meeting it at different times --/
theorem truck_speed (speed_A speed_B : ℝ) (time_A time_B : ℝ) : ℝ :=
  let distance_A := speed_A * time_A
  let distance_B := speed_B * time_B
  (distance_A - distance_B) / (time_B - time_A)

/-- Proof that the truck's speed is 52 km/h given the problem conditions --/
theorem truck_speed_is_52 :
  truck_speed 102 80 6 7 = 52 := by
  sorry

end NUMINAMATH_CALUDE_truck_speed_truck_speed_is_52_l2106_210653


namespace NUMINAMATH_CALUDE_bounded_g_given_bounded_f_l2106_210678

/-- Given real functions f and g defined on ℝ, satisfying certain conditions,
    prove that the absolute value of g is bounded by 1 for all real numbers. -/
theorem bounded_g_given_bounded_f (f g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x : ℝ, f x ≠ 0)
  (h3 : ∀ x : ℝ, |f x| ≤ 1) :
  ∀ y : ℝ, |g y| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_bounded_g_given_bounded_f_l2106_210678


namespace NUMINAMATH_CALUDE_x_values_in_A_l2106_210649

def A (x : ℝ) : Set ℝ := {-3, x + 2, x^2 - 4*x}

theorem x_values_in_A (x : ℝ) : 5 ∈ A x ↔ x = -1 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_values_in_A_l2106_210649


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2106_210610

theorem max_product_sum_300 : 
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧ 
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2106_210610


namespace NUMINAMATH_CALUDE_min_value_expression_l2106_210674

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : a * b = 2) :
  (a^2 + b^2 + 1) / (a - b) ≥ 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2106_210674


namespace NUMINAMATH_CALUDE_cycling_equation_correct_l2106_210688

/-- Represents the cycling speeds and time difference between two cyclists A and B. -/
structure CyclingProblem where
  distance : ℝ  -- Distance between points A and B
  speed_diff : ℝ  -- Speed difference between A and B
  time_diff : ℝ  -- Time difference of arrival (in hours)

/-- Checks if the given equation correctly represents the cycling problem. -/
def is_correct_equation (prob : CyclingProblem) (x : ℝ) : Prop :=
  prob.distance / x - prob.distance / (x + prob.speed_diff) = prob.time_diff

/-- The main theorem stating that the given equation correctly represents the cycling problem. -/
theorem cycling_equation_correct : 
  let prob : CyclingProblem := { distance := 30, speed_diff := 3, time_diff := 2/3 }
  ∀ x > 0, is_correct_equation prob x := by
  sorry

end NUMINAMATH_CALUDE_cycling_equation_correct_l2106_210688


namespace NUMINAMATH_CALUDE_gcd_lcm_a_b_l2106_210646

-- Define a and b
def a : Nat := 2 * 3 * 7
def b : Nat := 2 * 3 * 3 * 5

-- State the theorem
theorem gcd_lcm_a_b : Nat.gcd a b = 6 ∧ Nat.lcm a b = 630 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_a_b_l2106_210646


namespace NUMINAMATH_CALUDE_decimal_to_binary_13_l2106_210640

theorem decimal_to_binary_13 : (13 : ℕ) = 
  (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_13_l2106_210640


namespace NUMINAMATH_CALUDE_cage_cost_calculation_l2106_210623

def cat_toy_cost : ℝ := 10.22
def total_cost : ℝ := 21.95

theorem cage_cost_calculation : total_cost - cat_toy_cost = 11.73 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_calculation_l2106_210623


namespace NUMINAMATH_CALUDE_ratio_product_l2106_210613

theorem ratio_product (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  a * b * c / (d * e * f) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_product_l2106_210613


namespace NUMINAMATH_CALUDE_range_of_a_l2106_210614

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x + a * x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2106_210614


namespace NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l2106_210642

/-- A geometric sequence with positive terms where a₁, (1/2)a₃, 2a₂ form an arithmetic sequence -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  arithmetic : a 1 + 2 * a 2 = a 3

/-- The ratio of (a₁₁ + a₁₂) to (a₉ + a₁₀) equals 3 + 2√2 -/
theorem special_geometric_sequence_ratio 
  (seq : SpecialGeometricSequence) :
  (seq.a 11 + seq.a 12) / (seq.a 9 + seq.a 10) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l2106_210642


namespace NUMINAMATH_CALUDE_yellow_peaches_count_l2106_210662

theorem yellow_peaches_count (red green total : ℕ) 
  (h_red : red = 7)
  (h_green : green = 8)
  (h_total : total = 30)
  (h_sum : red + green + yellow = total) :
  yellow = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_count_l2106_210662


namespace NUMINAMATH_CALUDE_power_of_power_of_two_l2106_210682

theorem power_of_power_of_two :
  let a : ℕ := 2
  a^(a^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_of_two_l2106_210682


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l2106_210670

/-- For a normal distribution with mean 10.5 and standard deviation 1,
    the value that is exactly 2 standard deviations less than the mean is 8.5. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (hμ : μ = 10.5) (hσ : σ = 1) :
  μ - 2 * σ = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l2106_210670


namespace NUMINAMATH_CALUDE_certain_value_is_one_l2106_210637

theorem certain_value_is_one (w x : ℝ) (h1 : 13 = 13 * w / x) (h2 : w^2 = 1) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_is_one_l2106_210637


namespace NUMINAMATH_CALUDE_power_equality_l2106_210699

theorem power_equality (J : ℕ) (h : (32^4) * (4^4) = 2^J) : J = 28 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2106_210699


namespace NUMINAMATH_CALUDE_evaluate_expression_l2106_210627

theorem evaluate_expression : 3^13 / 3^3 + 2^3 = 59057 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2106_210627


namespace NUMINAMATH_CALUDE_apple_distribution_l2106_210665

theorem apple_distribution (total_apples : ℕ) (given_to_father : ℕ) (num_friends : ℕ) :
  total_apples = 55 →
  given_to_father = 10 →
  num_friends = 4 →
  (total_apples - given_to_father) % (num_friends + 1) = 0 →
  (total_apples - given_to_father) / (num_friends + 1) = 9 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l2106_210665


namespace NUMINAMATH_CALUDE_cylinder_volume_l2106_210686

/-- The volume of a cylinder with height 300 cm and circular base area of 9 square cm is 2700 cubic centimeters. -/
theorem cylinder_volume (h : ℝ) (base_area : ℝ) (volume : ℝ) 
  (h_val : h = 300)
  (base_area_val : base_area = 9)
  (volume_def : volume = base_area * h) : 
  volume = 2700 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l2106_210686


namespace NUMINAMATH_CALUDE_evaluate_expression_l2106_210652

theorem evaluate_expression : (10010 - 12 * 3) * 2 ^ 3 = 79792 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2106_210652


namespace NUMINAMATH_CALUDE_interest_rate_for_doubling_l2106_210635

/-- Represents the number of years required for the principal to double. -/
def years_to_double : ℝ := 10

/-- Theorem stating that if a principal doubles in 10 years due to simple interest,
    then the rate of interest is 10% per annum. -/
theorem interest_rate_for_doubling (P : ℝ) (P_pos : P > 0) :
  ∃ R : ℝ, R > 0 ∧ P + (P * R * years_to_double / 100) = 2 * P ∧ R = 10 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_for_doubling_l2106_210635


namespace NUMINAMATH_CALUDE_absolute_value_implication_l2106_210656

theorem absolute_value_implication (x : ℝ) : |x - 1| < 2 → x < 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_implication_l2106_210656


namespace NUMINAMATH_CALUDE_sons_shoveling_time_l2106_210696

/-- Proves that Wayne's son takes 21 hours to shovel the entire driveway alone,
    given that Wayne and his son together take 3 hours,
    and Wayne shovels 6 times as fast as his son. -/
theorem sons_shoveling_time (total_work : ℝ) (joint_time : ℝ) (wayne_speed_ratio : ℝ) :
  total_work > 0 →
  joint_time = 3 →
  wayne_speed_ratio = 6 →
  (total_work / joint_time) * (wayne_speed_ratio + 1) * 21 = total_work :=
by sorry

end NUMINAMATH_CALUDE_sons_shoveling_time_l2106_210696


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2106_210607

/-- An isosceles triangle with sides 4 and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 4 ∧ b = 9 ∧ c = 9 →  -- Two sides are 9, one side is 4
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 22 :=  -- Perimeter is 22
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2106_210607


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l2106_210690

/-- Given a regular pentagon and a rectangle with the same perimeter and the rectangle's length
    being twice its width, the ratio of the pentagon's side length to the rectangle's width is 6/5 -/
theorem pentagon_rectangle_ratio (p w l : ℝ) : 
  p > 0 → w > 0 → l > 0 →
  5 * p = 30 →  -- Pentagon perimeter
  2 * w + 2 * l = 30 →  -- Rectangle perimeter
  l = 2 * w →  -- Rectangle length is twice the width
  p / w = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l2106_210690


namespace NUMINAMATH_CALUDE_parabola_vertex_l2106_210645

/-- Define a parabola with equation y = (x+2)^2 + 3 -/
def parabola (x : ℝ) : ℝ := (x + 2)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-2, 3)

/-- Theorem: The vertex of the parabola y = (x+2)^2 + 3 is (-2, 3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2106_210645


namespace NUMINAMATH_CALUDE_antonella_toonies_l2106_210691

/-- Represents the number of coins of each type -/
structure CoinCount where
  loonies : ℕ
  toonies : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.loonies + 2 * coins.toonies

/-- Represents Antonella's coin situation -/
def antonellasCoins (coins : CoinCount) : Prop :=
  coins.loonies + coins.toonies = 10 ∧
  totalValue coins = 14

theorem antonella_toonies :
  ∃ (coins : CoinCount), antonellasCoins coins ∧ coins.toonies = 4 := by
  sorry

end NUMINAMATH_CALUDE_antonella_toonies_l2106_210691


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l2106_210692

theorem book_sale_loss_percentage 
  (total_cost : ℝ) 
  (cost_book1 : ℝ) 
  (gain_percentage : ℝ) :
  total_cost = 300 →
  cost_book1 = 175 →
  gain_percentage = 19 →
  let cost_book2 := total_cost - cost_book1
  let selling_price := cost_book2 * (1 + gain_percentage / 100)
  let loss_amount := cost_book1 - selling_price
  let loss_percentage := (loss_amount / cost_book1) * 100
  loss_percentage = 15 := by sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l2106_210692


namespace NUMINAMATH_CALUDE_power_equation_solution_l2106_210651

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26 → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2106_210651


namespace NUMINAMATH_CALUDE_tile_perimeter_increase_l2106_210602

/-- Represents a configuration of square tiles --/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration --/
def add_tiles (initial : TileConfiguration) (added : ℕ) : TileConfiguration :=
  { tiles := initial.tiles + added, perimeter := initial.perimeter + 3 }

/-- The theorem to be proved --/
theorem tile_perimeter_increase :
  ∃ (initial final : TileConfiguration),
    initial.tiles = 10 ∧
    initial.perimeter = 16 ∧
    final = add_tiles initial 3 ∧
    final.perimeter = 19 := by
  sorry

end NUMINAMATH_CALUDE_tile_perimeter_increase_l2106_210602


namespace NUMINAMATH_CALUDE_twelve_hash_six_l2106_210695

/-- The # operation for real numbers -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- Axioms for the # operation -/
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + 2 * s + 1

/-- The main theorem to prove -/
theorem twelve_hash_six : hash 12 6 = 272 := by
  sorry

end NUMINAMATH_CALUDE_twelve_hash_six_l2106_210695


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2106_210621

theorem negation_of_universal_proposition (a : ℝ) (h : 0 < a ∧ a < 1) :
  (¬ ∀ x : ℝ, x < 0 → a^x > 1) ↔ (∃ x₀ : ℝ, x₀ < 0 ∧ a^x₀ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2106_210621


namespace NUMINAMATH_CALUDE_line_intercept_ratio_l2106_210664

theorem line_intercept_ratio (b : ℝ) (u v : ℝ) (h : b ≠ 0) : 
  (5 * u + b = 0) → (3 * v + b = 0) → u / v = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_ratio_l2106_210664


namespace NUMINAMATH_CALUDE_digit_sequence_equality_l2106_210657

def A (n : ℕ) : ℕ := (10^n - 1) / 9

theorem digit_sequence_equality (n : ℕ) (hn : n > 0) :
  Real.sqrt ((10^n + 1) * A n - 2 * A n) = 3 * A n :=
sorry

end NUMINAMATH_CALUDE_digit_sequence_equality_l2106_210657


namespace NUMINAMATH_CALUDE_system_solution_l2106_210687

theorem system_solution (p q r s t : ℝ) :
  p^2 + q^2 + r^2 = 6 ∧ p * q - s^2 - t^2 = 3 →
  ((p = Real.sqrt 3 ∧ q = Real.sqrt 3 ∧ r = 0 ∧ s = 0 ∧ t = 0) ∨
   (p = -Real.sqrt 3 ∧ q = -Real.sqrt 3 ∧ r = 0 ∧ s = 0 ∧ t = 0)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2106_210687


namespace NUMINAMATH_CALUDE_fraction_comparison_l2106_210672

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a / b > (a + 1) / (b + 1) := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2106_210672


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l2106_210609

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l2106_210609


namespace NUMINAMATH_CALUDE_negation_of_all_students_punctual_l2106_210669

namespace NegationOfUniversalStatement

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (student : U → Prop)
variable (punctual : U → Prop)

-- State the theorem
theorem negation_of_all_students_punctual :
  (¬ ∀ x, student x → punctual x) ↔ (∃ x, student x ∧ ¬ punctual x) :=
sorry

end NegationOfUniversalStatement

end NUMINAMATH_CALUDE_negation_of_all_students_punctual_l2106_210669


namespace NUMINAMATH_CALUDE_rick_cards_count_l2106_210660

theorem rick_cards_count : ℕ := by
  -- Define the number of cards Rick kept
  let cards_kept : ℕ := 15

  -- Define the number of cards given to Miguel
  let cards_to_miguel : ℕ := 13

  -- Define the number of friends and cards given to each friend
  let num_friends : ℕ := 8
  let cards_per_friend : ℕ := 12

  -- Define the number of sisters and cards given to each sister
  let num_sisters : ℕ := 2
  let cards_per_sister : ℕ := 3

  -- Calculate the total number of cards
  let total_cards : ℕ := 
    cards_kept + 
    cards_to_miguel + 
    (num_friends * cards_per_friend) + 
    (num_sisters * cards_per_sister)

  -- Prove that the total number of cards is 130
  have h : total_cards = 130 := by sorry

  -- Return the result
  exact 130

end NUMINAMATH_CALUDE_rick_cards_count_l2106_210660


namespace NUMINAMATH_CALUDE_gas_experiment_values_l2106_210677

/-- Represents the state of a gas -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  temperature : ℝ

/-- Represents the change in gas state -/
structure GasStateChange where
  Δp : ℝ
  ΔV : ℝ

/-- Theorem stating the values of a₁ and a₂ for the given gas experiments -/
theorem gas_experiment_values (initialState : GasState) 
  (h_volume : initialState.volume = 1)
  (h_pressure : initialState.pressure = 10^5)
  (h_temperature : initialState.temperature = 300)
  (experiment1 : GasStateChange → Bool)
  (experiment2 : GasStateChange → Bool)
  (h_exp1 : ∀ change, experiment1 change ↔ change.Δp / change.ΔV = -10^5)
  (h_exp2 : ∀ change, experiment2 change ↔ change.Δp / change.ΔV = -1.4 * 10^5)
  (h_cooling1 : ∀ change, experiment1 change → 
    (change.ΔV > 0 → initialState.temperature > initialState.temperature + change.ΔV) ∧
    (change.ΔV < 0 → initialState.temperature > initialState.temperature - change.ΔV))
  (h_heating2 : ∀ change, experiment2 change → 
    (change.ΔV > 0 → initialState.temperature < initialState.temperature + change.ΔV) ∧
    (change.ΔV < 0 → initialState.temperature < initialState.temperature - change.ΔV)) :
  ∃ (a₁ a₂ : ℝ), a₁ = -10^5 ∧ a₂ = -1.4 * 10^5 := by
  sorry


end NUMINAMATH_CALUDE_gas_experiment_values_l2106_210677


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2106_210668

def in_quadrant_I_or_II (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∨ x < 0 ∧ y > 0

theorem points_in_quadrants_I_and_II (x y : ℝ) :
  y > 3 * x → y > 6 - x^2 → in_quadrant_I_or_II x y := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2106_210668


namespace NUMINAMATH_CALUDE_not_perfect_square_l2106_210625

theorem not_perfect_square : 
  (∃ x : ℕ, 6^2024 = x^2) ∧ 
  (∀ y : ℕ, 7^2025 ≠ y^2) ∧ 
  (∃ z : ℕ, 8^2026 = z^2) ∧ 
  (∃ w : ℕ, 9^2027 = w^2) ∧ 
  (∃ v : ℕ, 10^2028 = v^2) := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2106_210625


namespace NUMINAMATH_CALUDE_triangle_not_isosceles_l2106_210689

/-- A triangle with sides a, b, c is not isosceles if a, b, c are distinct -/
theorem triangle_not_isosceles (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : a ≠ b) (h₅ : b ≠ c) (h₆ : a ≠ c)
  (h₇ : a + b > c) (h₈ : b + c > a) (h₉ : a + c > b) :
  ¬(a = b ∨ b = c ∨ a = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_not_isosceles_l2106_210689


namespace NUMINAMATH_CALUDE_fruit_punch_total_l2106_210617

theorem fruit_punch_total (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) : 
  orange_punch = 4.5 →
  cherry_punch = 2 * orange_punch →
  apple_juice = cherry_punch - 1.5 →
  orange_punch + cherry_punch + apple_juice = 21 := by
  sorry

end NUMINAMATH_CALUDE_fruit_punch_total_l2106_210617


namespace NUMINAMATH_CALUDE_fraction_change_l2106_210622

theorem fraction_change (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : 
  (2*a) * (2*b) / (2*(2*a) + 2*b) = 2 * (a * b / (2*a + b)) :=
sorry

end NUMINAMATH_CALUDE_fraction_change_l2106_210622


namespace NUMINAMATH_CALUDE_dish_sets_budget_l2106_210626

theorem dish_sets_budget (total_budget : ℕ) (sets_at_20 : ℕ) (price_per_set : ℕ) :
  total_budget = 6800 →
  sets_at_20 = 178 →
  price_per_set = 20 →
  total_budget - (sets_at_20 * price_per_set) = 3240 :=
by sorry

end NUMINAMATH_CALUDE_dish_sets_budget_l2106_210626


namespace NUMINAMATH_CALUDE_initial_wings_count_l2106_210631

/-- The number of initially cooked chicken wings -/
def initial_wings : ℕ := sorry

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The number of additional wings cooked -/
def additional_wings : ℕ := 10

/-- The number of wings each person got -/
def wings_per_person : ℕ := 6

/-- Theorem stating that the number of initially cooked wings is 8 -/
theorem initial_wings_count : initial_wings = 8 := by sorry

end NUMINAMATH_CALUDE_initial_wings_count_l2106_210631


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l2106_210634

/-- A rectangle in a 2D coordinate system --/
structure Rectangle where
  x1 : ℝ
  x2 : ℝ
  y1 : ℝ
  y2 : ℝ

/-- The area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ :=
  |r.x2 - r.x1| * |r.y2 - r.y1|

/-- Theorem: If a rectangle with vertices (-8, y), (1, y), (1, -7), and (-8, -7) has an area of 72, then y = 1 --/
theorem rectangle_area_theorem (y : ℝ) :
  let r := Rectangle.mk (-8) 1 y (-7)
  r.area = 72 → y = 1 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_theorem_l2106_210634


namespace NUMINAMATH_CALUDE_unique_gcd_triplet_l2106_210676

-- Define the sets of possible values for x, y, and z
def X : Set ℕ := {6, 8, 12, 18, 24}
def Y : Set ℕ := {14, 20, 28, 44, 56}
def Z : Set ℕ := {5, 15, 18, 27, 42}

-- Define the theorem
theorem unique_gcd_triplet :
  ∃! (a b c x y z : ℕ),
    x ∈ X ∧ y ∈ Y ∧ z ∈ Z ∧
    x = Nat.gcd a b ∧
    y = Nat.gcd b c ∧
    z = Nat.gcd c a ∧
    x = 8 ∧ y = 14 ∧ z = 18 :=
by
  sorry

#check unique_gcd_triplet

end NUMINAMATH_CALUDE_unique_gcd_triplet_l2106_210676


namespace NUMINAMATH_CALUDE_gcd_3375_9180_l2106_210628

theorem gcd_3375_9180 : Nat.gcd 3375 9180 = 135 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3375_9180_l2106_210628


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2106_210624

theorem expand_and_simplify (x : ℝ) : (2*x + 6)*(x + 10) = 2*x^2 + 26*x + 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2106_210624


namespace NUMINAMATH_CALUDE_product_mod_500_l2106_210630

theorem product_mod_500 : (2367 * 1023) % 500 = 41 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_500_l2106_210630


namespace NUMINAMATH_CALUDE_distance_C2_C3_eq_sqrt_10m_l2106_210694

/-- Right triangle ABC with given side lengths -/
structure RightTriangleABC where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  right_angle : AB ^ 2 + AC ^ 2 = BC ^ 2
  AB_eq : AB = 80
  AC_eq : AC = 150
  BC_eq : BC = 170

/-- Inscribed circle C1 of triangle ABC -/
def C1 (t : RightTriangleABC) : Circle := sorry

/-- Line DE perpendicular to AC and tangent to C1 -/
def DE (t : RightTriangleABC) : Line := sorry

/-- Line FG perpendicular to AB and tangent to C1 -/
def FG (t : RightTriangleABC) : Line := sorry

/-- Inscribed circle C2 of triangle BDE -/
def C2 (t : RightTriangleABC) : Circle := sorry

/-- Inscribed circle C3 of triangle CFG -/
def C3 (t : RightTriangleABC) : Circle := sorry

/-- The distance between the centers of C2 and C3 -/
def distance_C2_C3 (t : RightTriangleABC) : ℝ := sorry

theorem distance_C2_C3_eq_sqrt_10m (t : RightTriangleABC) :
  distance_C2_C3 t = Real.sqrt (10 * 1057.6) := by sorry

end NUMINAMATH_CALUDE_distance_C2_C3_eq_sqrt_10m_l2106_210694


namespace NUMINAMATH_CALUDE_second_invoice_not_23_l2106_210683

def systematic_sampling (first : ℕ) : ℕ → ℕ := fun n => first + 10 * (n - 1)

theorem second_invoice_not_23 :
  ∀ first : ℕ, 1 ≤ first ∧ first ≤ 10 →
  systematic_sampling first 2 ≠ 23 := by
sorry

end NUMINAMATH_CALUDE_second_invoice_not_23_l2106_210683


namespace NUMINAMATH_CALUDE_extreme_values_and_three_roots_l2106_210611

/-- The function f(x) = x³ + ax² + bx + c -/
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_three_roots 
  (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, ∃ y₁ y₂ y₃, f a b c y₁ = 2*c ∧ f a b c y₂ = 2*c ∧ f a b c y₃ = 2*c ∧ y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃) →
  (f' a b 1 = 0 ∧ f' a b (-2/3) = 0) →
  (a = -1/2 ∧ b = -2 ∧ 1/2 ≤ c ∧ c < 22/27) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_three_roots_l2106_210611


namespace NUMINAMATH_CALUDE_number_problem_l2106_210620

theorem number_problem (x : ℚ) : (3 * x / 2 + 6 = 11) → x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2106_210620


namespace NUMINAMATH_CALUDE_book_count_l2106_210633

theorem book_count : ∃ (B : ℕ), 
  B > 0 ∧
  (2 * B) % 5 = 0 ∧  -- Two-fifths of books are reading books
  (3 * B) % 10 = 0 ∧ -- Three-tenths of books are math books
  (B * 3) / 10 - 1 = (B * 3) / 10 - (B * 3) % 10 / 10 - 1 ∧ -- Science books are one fewer than math books
  ((2 * B) / 5 + (3 * B) / 10 + ((3 * B) / 10 - 1) + 1 = B) ∧ -- Sum of all book types equals total
  B = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_count_l2106_210633


namespace NUMINAMATH_CALUDE_square_coverage_l2106_210655

theorem square_coverage (k n : ℕ) : k > 1 → (k ^ 2 = 2 ^ (n + 1) * n + 1) → (k = 7 ∧ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_square_coverage_l2106_210655


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2106_210619

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
  (h_arith : a 3 = 2 * (2 * a 1) - a 2) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2106_210619


namespace NUMINAMATH_CALUDE_parametric_curve_extrema_l2106_210648

open Real

theorem parametric_curve_extrema :
  let x : ℝ → ℝ := λ t ↦ 2 * (1 + cos t) * cos t
  let y : ℝ → ℝ := λ t ↦ 2 * (1 + cos t) * sin t
  let t_domain := {t : ℝ | 0 ≤ t ∧ t ≤ 2 * π}
  (∀ t ∈ t_domain, x t ≤ 4) ∧
  (∃ t ∈ t_domain, x t = 4) ∧
  (∀ t ∈ t_domain, x t ≥ -1/2) ∧
  (∃ t ∈ t_domain, x t = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_parametric_curve_extrema_l2106_210648


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2106_210671

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 667) ∧
  (has_no_prime_factors_less_than_20 667) ∧
  (∀ m : ℕ, m < 667 →
    ¬(is_composite m ∧ has_no_prime_factors_less_than_20 m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2106_210671


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l2106_210636

/-- Represents a triangle with side lengths a, b, and c --/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Generates the next triangle in the sequence based on the incircle tangency points --/
def nextTriangle (t : Triangle) : Triangle := sorry

/-- Checks if a triangle is valid (satisfies the triangle inequality) --/
def isValidTriangle (t : Triangle) : Bool := sorry

/-- The sequence of triangles starting from T₁ --/
def triangleSequence : List Triangle := sorry

/-- The last valid triangle in the sequence --/
def lastValidTriangle : Triangle := sorry

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℚ := sorry

theorem last_triangle_perimeter :
  let t₁ : Triangle := { a := 2011, b := 2012, c := 2013 }
  perimeter (lastValidTriangle) = 1509 / 128 := by sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l2106_210636


namespace NUMINAMATH_CALUDE_inconsistent_inventory_report_max_consistent_statements_no_more_than_three_consistent_l2106_210667

theorem inconsistent_inventory_report (n : ℕ) (h_n : n ≥ 1000) : 
  ¬(n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 2) :=
sorry

theorem max_consistent_statements : 
  ∃ (n : ℕ), n ≥ 1000 ∧ 
  ((n % 2 = 1 ∧ n % 3 = 1 ∧ n % 5 = 2) ∨
   (n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 2)) :=
sorry

theorem no_more_than_three_consistent (n : ℕ) (h_n : n ≥ 1000) :
  ¬∃ (a b c d : Bool), a ∧ b ∧ c ∧ d ∧
  (a → n % 2 = 1) ∧
  (b → n % 3 = 1) ∧
  (c → n % 4 = 2) ∧
  (d → n % 5 = 2) ∧
  (a.toNat + b.toNat + c.toNat + d.toNat > 3) :=
sorry

end NUMINAMATH_CALUDE_inconsistent_inventory_report_max_consistent_statements_no_more_than_three_consistent_l2106_210667


namespace NUMINAMATH_CALUDE_palace_visitors_l2106_210650

/-- The number of visitors to Buckingham Palace over two days -/
def total_visitors (day1 : ℕ) (day2 : ℕ) : ℕ := day1 + day2

/-- Theorem stating the total number of visitors over two days -/
theorem palace_visitors : total_visitors 583 246 = 829 := by
  sorry

end NUMINAMATH_CALUDE_palace_visitors_l2106_210650


namespace NUMINAMATH_CALUDE_probability_second_class_correct_l2106_210639

/-- The probability of selecting at least one second-class product when
    randomly choosing 4 products from a batch of 100 products containing
    90 first-class and 10 second-class products. -/
def probability_second_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ) : ℚ :=
  1 - (first_class / total) * ((first_class - 1) / (total - 1)) *
      ((first_class - 2) / (total - 2)) * ((first_class - 3) / (total - 3))

/-- The theorem stating that the probability of selecting at least one
    second-class product is correct for the given conditions. -/
theorem probability_second_class_correct :
  probability_second_class 100 90 10 4 = 1 - (90/100 * 89/99 * 88/98 * 87/97) :=
by sorry

end NUMINAMATH_CALUDE_probability_second_class_correct_l2106_210639


namespace NUMINAMATH_CALUDE_circle_condition_l2106_210681

/-- A circle in the xy-plane is represented by the equation x^2 + y^2 + Dx + Ey + F = 0,
    where D^2 + E^2 - 4F > 0 -/
def is_circle (D E F : ℝ) : Prop := D^2 + E^2 - 4*F > 0

/-- The equation x^2 + y^2 - 2x + 2k + 3 = 0 represents a circle -/
def our_equation_is_circle (k : ℝ) : Prop := is_circle (-2) 0 (2*k + 3)

theorem circle_condition (k : ℝ) : our_equation_is_circle k ↔ k < -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l2106_210681


namespace NUMINAMATH_CALUDE_log_sum_equality_l2106_210606

theorem log_sum_equality : 
  2 * Real.log 9 / Real.log 10 + 3 * Real.log 4 / Real.log 10 + 
  4 * Real.log 3 / Real.log 10 + 5 * Real.log 2 / Real.log 10 + 
  Real.log 16 / Real.log 10 = Real.log 215233856 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l2106_210606


namespace NUMINAMATH_CALUDE_parabola_single_intersection_l2106_210612

/-- A parabola with equation y = x^2 + 2x + k intersects the x-axis at only one point if and only if k = 1 -/
theorem parabola_single_intersection (k : ℝ) : 
  (∃! x, x^2 + 2*x + k = 0) ↔ k = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_l2106_210612


namespace NUMINAMATH_CALUDE_square_area_decrease_l2106_210629

theorem square_area_decrease (areaI areaIII areaII : ℝ) (decrease_percent : ℝ) :
  areaI = 18 * Real.sqrt 3 →
  areaIII = 50 * Real.sqrt 3 →
  areaII = 72 →
  decrease_percent = 20 →
  let side_length := Real.sqrt areaII
  let new_side_length := side_length * (1 - decrease_percent / 100)
  let new_area := new_side_length ^ 2
  (areaII - new_area) / areaII * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_decrease_l2106_210629


namespace NUMINAMATH_CALUDE_derivative_of_complex_function_l2106_210647

/-- The derivative of ln(4x - 1 + √(16x^2 - 8x + 2)) - √(16x^2 - 8x + 2) * arctan(4x - 1) -/
theorem derivative_of_complex_function (x : ℝ) 
  (h1 : 16 * x^2 - 8 * x + 2 ≥ 0) 
  (h2 : 4 * x - 1 + Real.sqrt (16 * x^2 - 8 * x + 2) > 0) :
  deriv (fun x => Real.log (4 * x - 1 + Real.sqrt (16 * x^2 - 8 * x + 2)) - 
    Real.sqrt (16 * x^2 - 8 * x + 2) * Real.arctan (4 * x - 1)) x = 
  (4 * (1 - 4 * x) / Real.sqrt (16 * x^2 - 8 * x + 2)) * Real.arctan (4 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_complex_function_l2106_210647
