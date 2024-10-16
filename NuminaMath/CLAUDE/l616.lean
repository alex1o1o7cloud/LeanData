import Mathlib

namespace NUMINAMATH_CALUDE_angle_value_in_plane_figure_l616_61673

theorem angle_value_in_plane_figure (x : ℝ) : 
  x > 0 ∧ 
  x + x + 140 = 360 → 
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_angle_value_in_plane_figure_l616_61673


namespace NUMINAMATH_CALUDE_focus_to_latus_rectum_distance_l616_61640

/-- A parabola with equation y^2 = 2px (p > 0) whose latus rectum is tangent to the circle (x-3)^2 + y^2 = 16 -/
structure TangentParabola where
  p : ℝ
  p_pos : p > 0
  latus_rectum_tangent : ∃ (x y : ℝ), y^2 = 2*p*x ∧ (x-3)^2 + y^2 = 16

/-- The distance from the focus of the parabola to the latus rectum is 2 -/
theorem focus_to_latus_rectum_distance (tp : TangentParabola) : tp.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_focus_to_latus_rectum_distance_l616_61640


namespace NUMINAMATH_CALUDE_inequality_solution_l616_61638

theorem inequality_solution (x : ℝ) : 
  (5 - 1 / (3 * x + 4) < 7) ↔ (x < -11/6 ∨ x > -4/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l616_61638


namespace NUMINAMATH_CALUDE_car_average_speed_l616_61693

/-- Given a car's speed for two consecutive hours, calculate its average speed. -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 100) (h2 : speed2 = 30) :
  (speed1 + speed2) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l616_61693


namespace NUMINAMATH_CALUDE_lcm_gcd_product_30_75_l616_61651

theorem lcm_gcd_product_30_75 : Nat.lcm 30 75 * Nat.gcd 30 75 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_30_75_l616_61651


namespace NUMINAMATH_CALUDE_min_students_with_both_traits_l616_61678

theorem min_students_with_both_traits (total : ℕ) (brown_eyes : ℕ) (lunch_box : ℕ) :
  total = 35 →
  brown_eyes = 18 →
  lunch_box = 25 →
  ∃ (both : ℕ), both ≥ 8 ∧
    ∀ (x : ℕ), x < 8 →
      total < brown_eyes + lunch_box - x :=
by sorry

end NUMINAMATH_CALUDE_min_students_with_both_traits_l616_61678


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l616_61674

def is_hyperbola (m : ℝ) : Prop :=
  (16 - m) * (9 - m) < 0

theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ 9 < m ∧ m < 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l616_61674


namespace NUMINAMATH_CALUDE_time_to_finish_book_l616_61646

/-- Calculates the time needed to finish a book given the current reading progress and reading speed. -/
theorem time_to_finish_book (total_pages reading_speed current_page : ℕ) 
  (h1 : total_pages = 210)
  (h2 : current_page = 90)
  (h3 : reading_speed = 30) : 
  (total_pages - current_page) / reading_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_time_to_finish_book_l616_61646


namespace NUMINAMATH_CALUDE_deposit_time_problem_l616_61628

/-- Proves that given the conditions of the problem, the deposit time is 3 years -/
theorem deposit_time_problem (initial_deposit : ℝ) (final_amount : ℝ) (final_amount_higher_rate : ℝ) 
  (h1 : initial_deposit = 8000)
  (h2 : final_amount = 10200)
  (h3 : final_amount_higher_rate = 10680) :
  ∃ (r : ℝ), 
    final_amount = initial_deposit + initial_deposit * (r / 100) * 3 ∧
    final_amount_higher_rate = initial_deposit + initial_deposit * ((r + 2) / 100) * 3 :=
sorry

end NUMINAMATH_CALUDE_deposit_time_problem_l616_61628


namespace NUMINAMATH_CALUDE_sine_equality_equivalence_l616_61660

theorem sine_equality_equivalence (α β : ℝ) : 
  (∃ k : ℤ, α = k * Real.pi + (-1)^k * β) ↔ Real.sin α = Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_sine_equality_equivalence_l616_61660


namespace NUMINAMATH_CALUDE_mike_found_four_more_seashells_l616_61663

/-- The number of seashells Mike initially found -/
def initial_seashells : ℝ := 6.0

/-- The total number of seashells Mike ended up with -/
def total_seashells : ℝ := 10

/-- The number of additional seashells Mike found -/
def additional_seashells : ℝ := total_seashells - initial_seashells

theorem mike_found_four_more_seashells : additional_seashells = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_mike_found_four_more_seashells_l616_61663


namespace NUMINAMATH_CALUDE_solve_linear_equation_l616_61642

theorem solve_linear_equation (x : ℝ) (h : 5*x - 7 = 15*x + 13) : 3*(x+10) = 24 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l616_61642


namespace NUMINAMATH_CALUDE_functions_inequality_l616_61611

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem functions_inequality (hf : f 0 = 0) 
  (hg : ∀ x y : ℝ, g (x - y) ≥ f x * f y + g x * g y) :
  ∀ x : ℝ, f x ^ 2008 + g x ^ 2008 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_functions_inequality_l616_61611


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l616_61699

theorem quadratic_form_ratio : 
  ∃ (d e : ℝ), (∀ x, x^2 + 800*x + 500 = (x + d)^2 + e) ∧ (e / d = -398.75) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l616_61699


namespace NUMINAMATH_CALUDE_total_balloons_count_l616_61661

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 41

/-- The total number of blue balloons Joan and Melanie have -/
def total_balloons : ℕ := joan_balloons + melanie_balloons

theorem total_balloons_count : total_balloons = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_count_l616_61661


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l616_61622

theorem fifteenth_student_age (total_students : Nat) (avg_age : Nat) (group1_size : Nat) (group1_avg : Nat) (group2_size : Nat) (group2_avg : Nat) :
  total_students = 15 →
  avg_age = 15 →
  group1_size = 3 →
  group1_avg = 14 →
  group2_size = 11 →
  group2_avg = 16 →
  (total_students * avg_age) - (group1_size * group1_avg + group2_size * group2_avg) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l616_61622


namespace NUMINAMATH_CALUDE_paint_needed_l616_61672

theorem paint_needed (total_needed : ℕ) (existing : ℕ) (newly_bought : ℕ) 
  (h1 : total_needed = 70)
  (h2 : existing = 36)
  (h3 : newly_bought = 23) :
  total_needed - (existing + newly_bought) = 11 := by
  sorry

end NUMINAMATH_CALUDE_paint_needed_l616_61672


namespace NUMINAMATH_CALUDE_least_possible_z_l616_61685

theorem least_possible_z (x y z : ℤ) : 
  Even x → Odd y → Odd z → y - x > 5 → (∀ w, Odd w → w - x ≥ 9 → z ≤ w) → z = 11 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_z_l616_61685


namespace NUMINAMATH_CALUDE_fraction_equality_l616_61650

theorem fraction_equality : (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l616_61650


namespace NUMINAMATH_CALUDE_sin_cos_identity_l616_61664

theorem sin_cos_identity : 
  Real.sin (17 * π / 180) * Real.sin (223 * π / 180) - 
  Real.sin (253 * π / 180) * Real.cos (43 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l616_61664


namespace NUMINAMATH_CALUDE_alpha_beta_equivalence_l616_61669

theorem alpha_beta_equivalence (α β : ℝ) :
  (α > β) ↔ (α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_equivalence_l616_61669


namespace NUMINAMATH_CALUDE_triangle_inequality_from_condition_l616_61619

theorem triangle_inequality_from_condition 
  (a b c : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (h : 5 * (a^2 + b^2 + c^2) < 6 * (a*b + b*c + c*a)) : 
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_condition_l616_61619


namespace NUMINAMATH_CALUDE_triangle_cutting_theorem_l616_61653

theorem triangle_cutting_theorem (x : ℝ) : 
  (∀ a b c : ℝ, a = 6 - x ∧ b = 8 - x ∧ c = 10 - x → a + b ≤ c) →
  x ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_cutting_theorem_l616_61653


namespace NUMINAMATH_CALUDE_a_bounded_by_two_l616_61639

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem a_bounded_by_two
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (a : ℝ)
  (h_ineq : ∀ x : ℝ, f (a * 2^x) - f (4^x + 1) ≤ 0) :
  -2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_a_bounded_by_two_l616_61639


namespace NUMINAMATH_CALUDE_distance_to_x_axis_on_ellipse_l616_61637

/-- The distance from a point on an ellipse to the x-axis, given specific conditions -/
theorem distance_to_x_axis_on_ellipse (x y : ℝ) : 
  (x^2 / 2 + y^2 / 6 = 1) →  -- Point (x, y) is on the ellipse
  (x * x + (y + 2) * (y - 2) = 0) →  -- Dot product condition
  |y| = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_on_ellipse_l616_61637


namespace NUMINAMATH_CALUDE_sqrt_undefined_for_positive_integer_l616_61603

theorem sqrt_undefined_for_positive_integer (x : ℕ+) :
  (¬ ∃ (y : ℝ), y ^ 2 = (x : ℝ) - 3) ↔ (x = 1 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_undefined_for_positive_integer_l616_61603


namespace NUMINAMATH_CALUDE_inscribed_cube_sphere_surface_area_l616_61604

theorem inscribed_cube_sphere_surface_area (cube_surface_area : ℝ) (sphere_surface_area : ℝ) :
  cube_surface_area = 6 →
  ∃ (cube_edge : ℝ) (sphere_radius : ℝ),
    cube_edge > 0 ∧
    sphere_radius > 0 ∧
    cube_surface_area = 6 * cube_edge^2 ∧
    sphere_radius = (cube_edge * Real.sqrt 3) / 2 ∧
    sphere_surface_area = 4 * Real.pi * sphere_radius^2 ∧
    sphere_surface_area = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_sphere_surface_area_l616_61604


namespace NUMINAMATH_CALUDE_current_rate_calculation_l616_61657

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 20) 
  (h2 : downstream_distance = 5.2) 
  (h3 : downstream_time = 0.2) : 
  ∃ (current_rate : ℝ), 
    current_rate = 6 ∧ 
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l616_61657


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_one_third_l616_61601

/-- The equation |x-3| = ax - 1 has two solutions if and only if a > 1/3 -/
theorem two_solutions_iff_a_gt_one_third (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (|x₁ - 3| = a * x₁ - 1) ∧ (|x₂ - 3| = a * x₂ - 1)) ↔ a > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_one_third_l616_61601


namespace NUMINAMATH_CALUDE_base5_sum_equality_l616_61618

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a base-5 representation to a natural number --/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base-5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem base5_sum_equality :
  addBase5 (toBase5 122) (toBase5 78) = toBase5 200 :=
sorry

end NUMINAMATH_CALUDE_base5_sum_equality_l616_61618


namespace NUMINAMATH_CALUDE_train_length_l616_61677

/-- Calculates the length of a train given its speed and time to cross a post -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 40 → time = 17.1 → speed * time * (5 / 18) = 190 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l616_61677


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l616_61648

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (h : x ≥ -1/3) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l616_61648


namespace NUMINAMATH_CALUDE_range_of_a_l616_61631

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) a, f x ∈ Set.Icc (-5 : ℝ) 4) ∧
  (∃ x₁ ∈ Set.Icc (-2 : ℝ) a, f x₁ = -5) ∧
  (∃ x₂ ∈ Set.Icc (-2 : ℝ) a, f x₂ = 4) →
  a ∈ Set.Icc (1 : ℝ) 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l616_61631


namespace NUMINAMATH_CALUDE_shortest_side_right_triangle_l616_61696

theorem shortest_side_right_triangle (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_a : a = 7) (h_b : b = 24) : 
  min a b = 7 := by sorry

end NUMINAMATH_CALUDE_shortest_side_right_triangle_l616_61696


namespace NUMINAMATH_CALUDE_solve_bucket_problem_l616_61659

def bucket_problem (b1 b2 b3 b4 b5 : ℕ) : Prop :=
  b1 = 11 ∧ b2 = 13 ∧ b3 = 12 ∧ b4 = 16 ∧ b5 = 10 →
  (b5 + b2 = 23) →
  (b1 + b3 + b4 = 39)

theorem solve_bucket_problem :
  ∀ b1 b2 b3 b4 b5 : ℕ, bucket_problem b1 b2 b3 b4 b5 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_bucket_problem_l616_61659


namespace NUMINAMATH_CALUDE_don_buys_150_from_shop_A_l616_61609

/-- The number of bottles Don buys from each shop -/
structure BottlePurchase where
  total : ℕ
  shopA : ℕ
  shopB : ℕ
  shopC : ℕ

/-- Don's bottle purchase satisfies the given conditions -/
def valid_purchase (p : BottlePurchase) : Prop :=
  p.total = 550 ∧ p.shopB = 180 ∧ p.shopC = 220 ∧ p.total = p.shopA + p.shopB + p.shopC

/-- Theorem: Don buys 150 bottles from Shop A -/
theorem don_buys_150_from_shop_A (p : BottlePurchase) (h : valid_purchase p) : p.shopA = 150 := by
  sorry

end NUMINAMATH_CALUDE_don_buys_150_from_shop_A_l616_61609


namespace NUMINAMATH_CALUDE_largest_cube_filling_box_l616_61629

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cube -/
def cubeVolume (edge : ℕ) : ℕ :=
  edge * edge * edge

/-- The main theorem about the largest cube that can fill the box -/
theorem largest_cube_filling_box (box : BoxDimensions) 
  (h_box : box = { length := 102, width := 255, height := 170 }) :
  let maxEdge := gcd3 box.length box.width box.height
  let numCubes := boxVolume box / cubeVolume maxEdge
  maxEdge = 17 ∧ numCubes = 900 := by sorry

end NUMINAMATH_CALUDE_largest_cube_filling_box_l616_61629


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_l616_61654

theorem sqrt_sum_difference (x : ℝ) : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) - Real.sqrt 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_l616_61654


namespace NUMINAMATH_CALUDE_final_ratio_is_16_to_9_l616_61624

/-- Represents the contents of a bin with peanuts and raisins -/
structure BinContents where
  peanuts : ℚ
  raisins : ℚ

/-- Removes an amount from the bin proportionally -/
def removeProportionally (bin : BinContents) (amount : ℚ) : BinContents :=
  let total := bin.peanuts + bin.raisins
  let peanutsProportion := bin.peanuts / total
  let raisinsProportion := bin.raisins / total
  { peanuts := bin.peanuts - (peanutsProportion * amount)
  , raisins := bin.raisins - (raisinsProportion * amount) }

/-- Adds an amount of raisins to the bin -/
def addRaisins (bin : BinContents) (amount : ℚ) : BinContents :=
  { peanuts := bin.peanuts, raisins := bin.raisins + amount }

/-- Theorem stating the final ratio of peanuts to raisins -/
theorem final_ratio_is_16_to_9 :
  let initial_bin : BinContents := { peanuts := 10, raisins := 0 }
  let after_first_operation := addRaisins { peanuts := initial_bin.peanuts - 2, raisins := 0 } 2
  let after_second_operation := addRaisins (removeProportionally after_first_operation 2) 2
  (after_second_operation.peanuts * 9 = after_second_operation.raisins * 16) := by
  sorry

end NUMINAMATH_CALUDE_final_ratio_is_16_to_9_l616_61624


namespace NUMINAMATH_CALUDE_f_2_equals_neg_1_l616_61607

-- Define the inverse function f⁻¹
def f_inv (x : ℝ) : ℝ := 1 + x^2

-- State the theorem
theorem f_2_equals_neg_1 :
  ∃ (f : ℝ → ℝ), (∀ x < 0, f_inv (f x) = x) ∧ (∀ y ≥ 1, f (f_inv y) = y) ∧ f 2 = -1 :=
sorry

end NUMINAMATH_CALUDE_f_2_equals_neg_1_l616_61607


namespace NUMINAMATH_CALUDE_wolves_hunt_in_five_days_l616_61682

/-- Calculates the number of days before wolves need to hunt again -/
def days_before_next_hunt (hunting_wolves : ℕ) (additional_wolves : ℕ) 
  (meat_per_wolf_per_day : ℕ) (meat_per_deer : ℕ) : ℕ :=
  let total_wolves := hunting_wolves + additional_wolves
  let daily_meat_requirement := total_wolves * meat_per_wolf_per_day
  let total_meat_from_hunt := hunting_wolves * meat_per_deer
  total_meat_from_hunt / daily_meat_requirement

theorem wolves_hunt_in_five_days : 
  days_before_next_hunt 4 16 8 200 = 5 := by sorry

end NUMINAMATH_CALUDE_wolves_hunt_in_five_days_l616_61682


namespace NUMINAMATH_CALUDE_cn_length_l616_61635

/-- Right-angled triangle with squares on legs -/
structure RightTriangleWithSquares where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  -- Conditions
  right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 4
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 1
  square_acde : (D.1 - A.1) = (E.1 - C.1) ∧ (D.2 - A.2) = (E.2 - C.2) ∧
                 (D.1 - A.1) * (E.1 - C.1) + (D.2 - A.2) * (E.2 - C.2) = 0
  square_bcfg : (F.1 - B.1) = (G.1 - C.1) ∧ (F.2 - B.2) = (G.2 - C.2) ∧
                 (F.1 - B.1) * (G.1 - C.1) + (F.2 - B.2) * (G.2 - C.2) = 0
  m_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  n_on_cm : (N.2 - C.2) * (M.1 - C.1) = (N.1 - C.1) * (M.2 - C.2)
  n_on_df : (N.2 - D.2) * (F.1 - D.1) = (N.1 - D.1) * (F.2 - D.2)

/-- The length of CN is √17 -/
theorem cn_length (t : RightTriangleWithSquares) : 
  Real.sqrt ((t.N.1 - t.C.1)^2 + (t.N.2 - t.C.2)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_cn_length_l616_61635


namespace NUMINAMATH_CALUDE_product_inequality_l616_61690

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  let M := (1 - 1/a) * (1 - 1/b) * (1 - 1/c)
  M ≤ -8 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l616_61690


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l616_61668

theorem simplify_sqrt_fraction : 
  (Real.sqrt ((7:ℝ)^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l616_61668


namespace NUMINAMATH_CALUDE_special_function_property_l616_61691

/-- A function satisfying the given condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f (f y))^2)

/-- The main theorem to be proved -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
sorry

end NUMINAMATH_CALUDE_special_function_property_l616_61691


namespace NUMINAMATH_CALUDE_company_workforce_l616_61697

theorem company_workforce (initial_total : ℕ) : 
  (initial_total * 60 = initial_total * 100 * 60 / 100) →
  ((initial_total + 24) * 55 = (initial_total * 60) * 100 / (initial_total + 24)) →
  (initial_total + 24 = 288) := by
sorry

end NUMINAMATH_CALUDE_company_workforce_l616_61697


namespace NUMINAMATH_CALUDE_complex_number_location_l616_61621

theorem complex_number_location :
  let z : ℂ := (2 - I) / I
  (z.re < 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l616_61621


namespace NUMINAMATH_CALUDE_sams_trip_length_l616_61626

theorem sams_trip_length (total : ℚ) 
  (h1 : total / 4 + 24 + total / 6 = total) : total = 288 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sams_trip_length_l616_61626


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l616_61623

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧ 
  n % 2 = 1 ∧ n % 3 = 2 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l616_61623


namespace NUMINAMATH_CALUDE_solution_y_b_percentage_l616_61636

-- Define the solutions and their compositions
def solution_x_a : ℝ := 0.3
def solution_x_b : ℝ := 0.7
def solution_y_a : ℝ := 0.4

-- Define the mixture composition
def mixture_x : ℝ := 0.8
def mixture_y : ℝ := 0.2
def mixture_a : ℝ := 0.32

-- Theorem to prove
theorem solution_y_b_percentage : 
  solution_x_a + solution_x_b = 1 →
  mixture_x + mixture_y = 1 →
  mixture_x * solution_x_a + mixture_y * solution_y_a = mixture_a →
  1 - solution_y_a = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_solution_y_b_percentage_l616_61636


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l616_61617

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l616_61617


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l616_61634

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 73/15 ∧ B = 17/15 ∧
  ∀ (x : ℚ), x ≠ 12 → x ≠ -3 →
    (6*x + 1) / (x^2 - 9*x - 36) = A / (x - 12) + B / (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l616_61634


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_of_polynomial_l616_61681

theorem pure_imaginary_solutions_of_polynomial (x : ℂ) :
  (x^4 - 4*x^3 + 6*x^2 - 40*x - 48 = 0) ∧ (∃ k : ℝ, x = k * Complex.I) ↔
  x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_of_polynomial_l616_61681


namespace NUMINAMATH_CALUDE_number_times_five_equals_hundred_l616_61670

theorem number_times_five_equals_hundred (x : ℝ) : 5 * x = 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_times_five_equals_hundred_l616_61670


namespace NUMINAMATH_CALUDE_sock_problem_l616_61605

theorem sock_problem (n : ℕ) : n ≤ 8 → (Nat.choose 8 n = 56) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sock_problem_l616_61605


namespace NUMINAMATH_CALUDE_three_student_committees_l616_61687

theorem three_student_committees (n k : ℕ) (hn : n = 10) (hk : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_student_committees_l616_61687


namespace NUMINAMATH_CALUDE_total_bars_is_504_l616_61686

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 18

/-- The number of chocolate bars in each small box -/
def bars_per_small_box : ℕ := 28

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * bars_per_small_box

/-- Theorem: The total number of chocolate bars in the large box is 504 -/
theorem total_bars_is_504 : total_chocolate_bars = 504 := by
  sorry

end NUMINAMATH_CALUDE_total_bars_is_504_l616_61686


namespace NUMINAMATH_CALUDE_tennis_ball_order_l616_61666

/-- The number of tennis balls originally ordered by a sports retailer -/
def original_order : ℕ := 288

/-- The number of extra yellow balls sent by mistake -/
def extra_yellow_balls : ℕ := 90

/-- The ratio of white balls to yellow balls after the error -/
def final_ratio : Rat := 8 / 13

theorem tennis_ball_order :
  ∃ (white yellow : ℕ),
    -- The retailer ordered equal numbers of white and yellow tennis balls
    white = yellow ∧
    -- The total original order
    white + yellow = original_order ∧
    -- After the error, the ratio of white to yellow balls is 8/13
    (white : Rat) / ((yellow : Rat) + extra_yellow_balls) = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_tennis_ball_order_l616_61666


namespace NUMINAMATH_CALUDE_induction_contrapositive_l616_61600

theorem induction_contrapositive (P : ℕ → Prop) :
  (∀ k : ℕ, k > 0 → (P k → P (k + 1))) →
  (¬ P 4) →
  (¬ P 3) :=
by sorry

end NUMINAMATH_CALUDE_induction_contrapositive_l616_61600


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l616_61612

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- State the theorem
theorem opposite_of_negative_five :
  opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l616_61612


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_81_l616_61643

theorem factor_x_squared_minus_81 (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_81_l616_61643


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l616_61652

/-- A function f(x) = ax^2 - x - 1 has exactly one root if and only if a = 0 or a = -1/4 -/
theorem unique_root_quadratic (a : ℝ) : 
  (∃! x, a * x^2 - x - 1 = 0) ↔ (a = 0 ∨ a = -1/4) := by
sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l616_61652


namespace NUMINAMATH_CALUDE_wage_decrease_percentage_l616_61614

theorem wage_decrease_percentage (W : ℝ) (P : ℝ) : 
  W > 0 →  -- Wages are positive
  0.20 * (W - (P / 100) * W) = 0.50 * (0.30 * W) → 
  P = 25 :=
by sorry

end NUMINAMATH_CALUDE_wage_decrease_percentage_l616_61614


namespace NUMINAMATH_CALUDE_valid_paths_count_l616_61662

/-- Represents the number of paths on a complete 9x3 grid -/
def total_paths : ℕ := 220

/-- Represents the number of paths through each forbidden segment -/
def forbidden_segment_paths : ℕ := 70

/-- Represents the number of forbidden segments -/
def num_forbidden_segments : ℕ := 2

/-- Theorem stating the number of valid paths on the grid with forbidden segments -/
theorem valid_paths_count : 
  total_paths - (forbidden_segment_paths * num_forbidden_segments) = 80 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l616_61662


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l616_61671

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 864 →
  (∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧
    volume = side_length^3) →
  volume = 1728 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l616_61671


namespace NUMINAMATH_CALUDE_problem_solution_l616_61694

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Define the theorem
theorem problem_solution :
  (∀ a : ℝ, a > 0) →
  (∀ x : ℝ, p x 1 ∧ q x → 2 < x ∧ x < 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x a) → 1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l616_61694


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l616_61680

/-- The area of a rectangular yard with a square cut out -/
def fenced_area (length width cut_size : ℝ) : ℝ :=
  length * width - cut_size * cut_size

/-- Theorem: The area of a 20-foot by 18-foot rectangular region with a 4-foot by 4-foot square cut out is 344 square feet -/
theorem fenced_area_calculation :
  fenced_area 20 18 4 = 344 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l616_61680


namespace NUMINAMATH_CALUDE_probability_of_no_mismatch_l616_61695

/-- The number of red socks -/
def num_red_socks : ℕ := 4

/-- The number of blue socks -/
def num_blue_socks : ℕ := 4

/-- The total number of socks -/
def total_socks : ℕ := num_red_socks + num_blue_socks

/-- The number of pairs to be formed -/
def num_pairs : ℕ := total_socks / 2

/-- The number of ways to divide red socks into pairs -/
def red_pairings : ℕ := (Nat.choose num_red_socks 2) / 2

/-- The number of ways to divide blue socks into pairs -/
def blue_pairings : ℕ := (Nat.choose num_blue_socks 2) / 2

/-- The total number of favorable pairings -/
def favorable_pairings : ℕ := red_pairings * blue_pairings

/-- The total number of possible pairings -/
def total_pairings : ℕ := (Nat.factorial total_socks) / ((Nat.factorial 2)^num_pairs * Nat.factorial num_pairs)

/-- The probability of no mismatched pairs -/
def probability_no_mismatch : ℚ := favorable_pairings / total_pairings

theorem probability_of_no_mismatch : probability_no_mismatch = 3 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_no_mismatch_l616_61695


namespace NUMINAMATH_CALUDE_same_combination_probability_is_correct_l616_61645

def jar_candies : ℕ × ℕ := (12, 8)

def total_candies : ℕ := jar_candies.1 + jar_candies.2

def same_combination_probability : ℚ :=
  let terry_picks := Nat.choose total_candies 2
  let mary_picks := Nat.choose (total_candies - 2) 2
  let both_red := (Nat.choose jar_candies.1 2 * Nat.choose (jar_candies.1 - 2) 2) / (terry_picks * mary_picks)
  let both_blue := (Nat.choose jar_candies.2 2 * Nat.choose (jar_candies.2 - 2) 2) / (terry_picks * mary_picks)
  let mixed := (Nat.choose jar_candies.1 1 * Nat.choose jar_candies.2 1 * 
                Nat.choose (jar_candies.1 - 1) 1 * Nat.choose (jar_candies.2 - 1) 1) / 
               (terry_picks * mary_picks)
  both_red + both_blue + mixed

theorem same_combination_probability_is_correct : 
  same_combination_probability = 143 / 269 := by
  sorry

end NUMINAMATH_CALUDE_same_combination_probability_is_correct_l616_61645


namespace NUMINAMATH_CALUDE_unique_solution_l616_61625

theorem unique_solution : ∃! n : ℝ, 7 * n - 15 = 2 * n + 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l616_61625


namespace NUMINAMATH_CALUDE_train_length_train_length_problem_l616_61658

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + person_speed
  let relative_speed_mps := relative_speed * 1000 / 3600
  relative_speed_mps * passing_time

/-- Proof that a train with speed 56 km/hr passing a man running at 6 km/hr in the opposite direction in 6.386585847325762 seconds has a length of approximately 110 meters. -/
theorem train_length_problem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_length 56 6 6.386585847325762 - 110| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_problem_l616_61658


namespace NUMINAMATH_CALUDE_at_least_two_fever_probability_l616_61608

def vaccine_fever_prob : ℝ := 0.80

def num_people : ℕ := 3

def at_least_two_fever_prob : ℝ := 
  (Nat.choose num_people 2) * (vaccine_fever_prob ^ 2) * (1 - vaccine_fever_prob) +
  vaccine_fever_prob ^ num_people

theorem at_least_two_fever_probability :
  at_least_two_fever_prob = 0.896 := by sorry

end NUMINAMATH_CALUDE_at_least_two_fever_probability_l616_61608


namespace NUMINAMATH_CALUDE_pure_imaginary_product_imaginary_part_quotient_l616_61613

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := 2 + m * Complex.I

-- Part 1
theorem pure_imaginary_product (m : ℝ) :
  (z₁ m * z₂ m).re = 0 → m = 0 :=
sorry

-- Part 2
theorem imaginary_part_quotient (m : ℝ) :
  z₁ m ^ 2 - 2 * z₁ m + 2 = 0 →
  (z₂ m / z₁ m).im = -1/2 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_imaginary_part_quotient_l616_61613


namespace NUMINAMATH_CALUDE_larger_integer_value_l616_61679

theorem larger_integer_value (a b : ℕ+) (h1 : (a : ℚ) / (b : ℚ) = 3 / 2) (h2 : (a : ℕ) * b = 108) : 
  a = ⌊9 * Real.sqrt 2⌋ := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l616_61679


namespace NUMINAMATH_CALUDE_max_additional_plates_l616_61627

/-- Represents a set of letters for license plates --/
def LetterSet := List Char

/-- Calculate the number of license plates given three sets of letters --/
def calculatePlates (set1 set2 set3 : LetterSet) : Nat :=
  set1.length * set2.length * set3.length

/-- The initial sets of letters --/
def initialSet1 : LetterSet := ['C', 'H', 'L', 'P', 'R', 'S']
def initialSet2 : LetterSet := ['A', 'I', 'O', 'U']
def initialSet3 : LetterSet := ['D', 'M', 'N', 'T', 'V']

/-- The number of new letters to be added --/
def newLettersCount : Nat := 3

/-- Theorem: The maximum number of additional license plates is 96 --/
theorem max_additional_plates :
  (∀ newSet1 newSet2 newSet3 : LetterSet,
    newSet1.length + newSet2.length + newSet3.length = initialSet1.length + initialSet2.length + initialSet3.length + newLettersCount →
    calculatePlates newSet1 newSet2 newSet3 - calculatePlates initialSet1 initialSet2 initialSet3 ≤ 96) ∧
  (∃ newSet1 newSet2 newSet3 : LetterSet,
    newSet1.length + newSet2.length + newSet3.length = initialSet1.length + initialSet2.length + initialSet3.length + newLettersCount ∧
    calculatePlates newSet1 newSet2 newSet3 - calculatePlates initialSet1 initialSet2 initialSet3 = 96) := by
  sorry


end NUMINAMATH_CALUDE_max_additional_plates_l616_61627


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l616_61675

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  P.1 ≥ 0 →
  P.2 ≥ 0 →
  P.1^2 / a^2 - P.2^2 / b^2 = 1 →
  P.1^2 + P.2^2 = a^2 + b^2 →
  F₁.1 < 0 →
  F₂.1 > 0 →
  F₁.2 = 0 →
  F₂.2 = 0 →
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 3 * Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) →
  Real.sqrt ((F₂.1 - F₁.1)^2 / (2*a)^2) = Real.sqrt 10 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l616_61675


namespace NUMINAMATH_CALUDE_rational_combination_equals_24_l616_61667

theorem rational_combination_equals_24 :
  ∃ (f : List ℚ → ℚ),
    f [-1, -2, -3, -4] = 24 ∧
    (∀ x y z w, f [x, y, z, w] = ((x + y + z) * w) ∨
                f [x, y, z, w] = ((x + y + z) / w) ∨
                f [x, y, z, w] = ((x + y - z) * w) ∨
                f [x, y, z, w] = ((x + y - z) / w) ∨
                f [x, y, z, w] = ((x - y + z) * w) ∨
                f [x, y, z, w] = ((x - y + z) / w) ∨
                f [x, y, z, w] = ((x - y - z) * w) ∨
                f [x, y, z, w] = ((x - y - z) / w)) :=
by
  sorry

end NUMINAMATH_CALUDE_rational_combination_equals_24_l616_61667


namespace NUMINAMATH_CALUDE_consecutive_digits_count_l616_61684

theorem consecutive_digits_count : ∃ (m n : ℕ), 
  (10^(m-1) < 2^2020 ∧ 2^2020 < 10^m) ∧
  (10^(n-1) < 5^2020 ∧ 5^2020 < 10^n) ∧
  m + n = 2021 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_digits_count_l616_61684


namespace NUMINAMATH_CALUDE_log_ratio_problem_l616_61665

theorem log_ratio_problem (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h_log : Real.log p / Real.log 16 = Real.log q / Real.log 20 ∧ 
           Real.log p / Real.log 16 = Real.log (p + q) / Real.log 25) : 
  p / q = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_problem_l616_61665


namespace NUMINAMATH_CALUDE_initial_minutes_plan_a_l616_61602

/-- Represents the cost in dollars for a call under Plan A -/
def costPlanA (initialMinutes : ℕ) (totalMinutes : ℕ) : ℚ :=
  0.60 + 0.06 * (totalMinutes - initialMinutes)

/-- Represents the cost in dollars for a call under Plan B -/
def costPlanB (minutes : ℕ) : ℚ :=
  0.08 * minutes

theorem initial_minutes_plan_a : ∃ (x : ℕ), 
  (∀ (m : ℕ), m ≥ x → costPlanA x m = costPlanB m) ∧
  (costPlanA x 18 = costPlanB 18) ∧
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_minutes_plan_a_l616_61602


namespace NUMINAMATH_CALUDE_tracy_candies_problem_l616_61606

theorem tracy_candies_problem : ∃ (initial : ℕ), 
  (initial % 4 = 0) ∧ 
  (initial % 6 = 0) ∧ 
  (∃ (brother_took : ℕ), 
    (2 ≤ brother_took) ∧ 
    (brother_took ≤ 6) ∧ 
    (initial * 3 / 4 * 2 / 3 - 40 - brother_took = 5)) ∧ 
  initial = 96 := by
sorry

end NUMINAMATH_CALUDE_tracy_candies_problem_l616_61606


namespace NUMINAMATH_CALUDE_village_population_equality_second_village_initial_population_l616_61692

/-- The initial population of Village X -/
def initial_pop_X : ℕ := 68000

/-- The yearly decrease in population of Village X -/
def decrease_rate_X : ℕ := 1200

/-- The yearly increase in population of the second village -/
def increase_rate_Y : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years : ℕ := 13

/-- The initial population of the second village -/
def initial_pop_Y : ℕ := 42000

theorem village_population_equality :
  initial_pop_X - years * decrease_rate_X = initial_pop_Y + years * increase_rate_Y :=
by sorry

theorem second_village_initial_population :
  initial_pop_Y = 42000 :=
by sorry

end NUMINAMATH_CALUDE_village_population_equality_second_village_initial_population_l616_61692


namespace NUMINAMATH_CALUDE_max_value_of_expression_l616_61633

theorem max_value_of_expression (a b c d e f g h k : Int) 
  (ha : a = 1 ∨ a = -1) (hb : b = 1 ∨ b = -1) (hc : c = 1 ∨ c = -1)
  (hd : d = 1 ∨ d = -1) (he : e = 1 ∨ e = -1) (hf : f = 1 ∨ f = -1)
  (hg : g = 1 ∨ g = -1) (hh : h = 1 ∨ h = -1) (hk : k = 1 ∨ k = -1) :
  (∃ (a' b' c' d' e' f' g' h' k' : Int),
    (a' = 1 ∨ a' = -1) ∧ (b' = 1 ∨ b' = -1) ∧ (c' = 1 ∨ c' = -1) ∧
    (d' = 1 ∨ d' = -1) ∧ (e' = 1 ∨ e' = -1) ∧ (f' = 1 ∨ f' = -1) ∧
    (g' = 1 ∨ g' = -1) ∧ (h' = 1 ∨ h' = -1) ∧ (k' = 1 ∨ k' = -1) ∧
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' = 4) ∧
  (∀ (a' b' c' d' e' f' g' h' k' : Int),
    (a' = 1 ∨ a' = -1) → (b' = 1 ∨ b' = -1) → (c' = 1 ∨ c' = -1) →
    (d' = 1 ∨ d' = -1) → (e' = 1 ∨ e' = -1) → (f' = 1 ∨ f' = -1) →
    (g' = 1 ∨ g' = -1) → (h' = 1 ∨ h' = -1) → (k' = 1 ∨ k' = -1) →
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l616_61633


namespace NUMINAMATH_CALUDE_square_of_negative_two_m_cubed_l616_61656

theorem square_of_negative_two_m_cubed (m : ℝ) : (-2 * m^3)^2 = 4 * m^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_m_cubed_l616_61656


namespace NUMINAMATH_CALUDE_snack_packs_needed_l616_61610

def trail_mix_pack_size : ℕ := 6
def granola_bar_pack_size : ℕ := 8
def fruit_cup_pack_size : ℕ := 4
def total_people : ℕ := 18

def min_packs_needed (pack_size : ℕ) (people : ℕ) : ℕ :=
  (people + pack_size - 1) / pack_size

theorem snack_packs_needed :
  (min_packs_needed trail_mix_pack_size total_people = 3) ∧
  (min_packs_needed granola_bar_pack_size total_people = 3) ∧
  (min_packs_needed fruit_cup_pack_size total_people = 5) :=
by sorry

end NUMINAMATH_CALUDE_snack_packs_needed_l616_61610


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l616_61620

/-- Given a geometric sequence with first term a₁ = 2, 
    the minimum value of 3a₂ + 6a₃ is -3/2. -/
theorem min_value_geometric_sequence (r : ℝ) : 
  let a₁ : ℝ := 2
  let a₂ : ℝ := a₁ * r
  let a₃ : ℝ := a₂ * r
  3 * a₂ + 6 * a₃ ≥ -3/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l616_61620


namespace NUMINAMATH_CALUDE_johns_age_l616_61641

/-- Given that John is 30 years younger than his dad and the sum of their ages is 80 years, 
    prove that John is 25 years old. -/
theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l616_61641


namespace NUMINAMATH_CALUDE_complement_equivalence_l616_61649

def U (a : ℝ) := {x : ℕ | 0 < x ∧ x ≤ ⌊a⌋}
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {4, 5, 6}

theorem complement_equivalence (a : ℝ) :
  (6 ≤ a ∧ a < 7) ↔ (U a \ P = Q) :=
sorry

end NUMINAMATH_CALUDE_complement_equivalence_l616_61649


namespace NUMINAMATH_CALUDE_base_height_calculation_l616_61689

/-- Given a sculpture height and total height, calculate the base height -/
theorem base_height_calculation (sculpture_height_feet : ℚ) (sculpture_height_inches : ℚ) (total_height : ℚ) : 
  sculpture_height_feet = 2 ∧ 
  sculpture_height_inches = 10 ∧ 
  total_height = 3.6666666666666665 →
  total_height - (sculpture_height_feet + sculpture_height_inches / 12) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_base_height_calculation_l616_61689


namespace NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_double_root_condition_l616_61688

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ (a * x^2 + b * x + c = 0) ∧ (a * y^2 + b * y + c = 0) ∧ (x = 2 * y ∨ y = 2 * x)

/-- Theorem for the first part of the problem -/
theorem first_equation_is_double_root : is_double_root_equation 1 (-6) 8 := by sorry

/-- Theorem for the second part of the problem -/
theorem second_equation_double_root_condition (n : ℝ) : 
  is_double_root_equation 1 (-8 - n) (8 * n) → n = 4 ∨ n = 16 := by sorry

end NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_double_root_condition_l616_61688


namespace NUMINAMATH_CALUDE_parabola_coefficient_l616_61683

/-- A quadratic function with vertex form (x - h)^2 where h is the x-coordinate of the vertex -/
def quadratic_vertex_form (a : ℝ) (h : ℝ) (x : ℝ) : ℝ := a * (x - h)^2

theorem parabola_coefficient (f : ℝ → ℝ) (h : ℝ) (a : ℝ) :
  (∀ x, f x = quadratic_vertex_form a h x) →
  f 5 = -36 →
  h = 2 →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l616_61683


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_parallel_planes_perpendicular_implication_l616_61644

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem 1
theorem parallel_perpendicular_implication
  (m n : Line) (α : Plane)
  (h1 : parallel_lines m n)
  (h2 : perpendicular m α) :
  perpendicular n α :=
sorry

-- Theorem 2
theorem parallel_planes_perpendicular_implication
  (m n : Line) (α β : Plane)
  (h1 : parallel_planes α β)
  (h2 : parallel_lines m n)
  (h3 : perpendicular m α) :
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_parallel_planes_perpendicular_implication_l616_61644


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l616_61630

theorem complex_arithmetic_equality : (18 * 23 - 24 * 17) / 3 + 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l616_61630


namespace NUMINAMATH_CALUDE_painting_selection_ways_l616_61632

theorem painting_selection_ways (oil_paintings : ℕ) (chinese_paintings : ℕ) (watercolor_paintings : ℕ)
  (h1 : oil_paintings = 3)
  (h2 : chinese_paintings = 4)
  (h3 : watercolor_paintings = 5) :
  oil_paintings + chinese_paintings + watercolor_paintings = 12 := by
  sorry

end NUMINAMATH_CALUDE_painting_selection_ways_l616_61632


namespace NUMINAMATH_CALUDE_no_partition_sum_product_l616_61655

theorem no_partition_sum_product : ¬ ∃ (x y : ℕ), 
  1 ≤ x ∧ x ≤ 15 ∧ 1 ≤ y ∧ y ≤ 15 ∧ x ≠ y ∧
  x * y = (List.range 16).sum - x - y := by
  sorry

end NUMINAMATH_CALUDE_no_partition_sum_product_l616_61655


namespace NUMINAMATH_CALUDE_major_product_l616_61698

/-- Given a class with accounting (p), finance (q), marketing (r), and strategy (s) majors,
    prove that if there are 3 accounting majors and the product of all majors is 1365,
    then the product of finance, marketing, and strategy majors is 455. -/
theorem major_product (p q r s : ℕ) (h1 : p = 3) (h2 : p * q * r * s = 1365) :
  q * r * s = 455 := by
  sorry

end NUMINAMATH_CALUDE_major_product_l616_61698


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l616_61676

-- Define the fuel efficiency of the car
def fuel_efficiency : ℝ := 32

-- Define the distance the car can travel
def distance : ℝ := 368

-- Define the total cost of gas
def total_cost : ℝ := 46

-- Theorem to prove the cost of gas per gallon
theorem gas_cost_per_gallon :
  (total_cost / (distance / fuel_efficiency)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l616_61676


namespace NUMINAMATH_CALUDE_equation_solution_l616_61615

theorem equation_solution : ∃! x : ℚ, (3 / 4 : ℚ) + 1 / x = 7 / 8 :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l616_61615


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l616_61647

theorem ceiling_floor_sum (x : ℝ) : 
  ⌈x⌉ - ⌊x⌋ = 0 → ⌈x⌉ + ⌊x⌋ = 2*x := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l616_61647


namespace NUMINAMATH_CALUDE_point_e_satisfies_conditions_l616_61616

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Theorem: Point E(0, 0) satisfies the area ratio conditions in quadrilateral ABCD -/
theorem point_e_satisfies_conditions (A B C D E : Point) 
  (hA : A = ⟨-2, -4⟩) (hB : B = ⟨-2, 3⟩) (hC : C = ⟨4, 6⟩) (hD : D = ⟨4, -1⟩) (hE : E = ⟨0, 0⟩) :
  triangleArea E A B / triangleArea E C D = 1 / 2 ∧
  triangleArea E A D / triangleArea E B C = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_point_e_satisfies_conditions_l616_61616
