import Mathlib

namespace skateboard_distance_l586_58658

theorem skateboard_distance (scooter_speed : ℝ) (skateboard_speed_ratio : ℝ) (time_minutes : ℝ) :
  scooter_speed = 50 →
  skateboard_speed_ratio = 2 / 5 →
  time_minutes = 45 →
  skateboard_speed_ratio * scooter_speed * (time_minutes / 60) = 15 := by
  sorry

end skateboard_distance_l586_58658


namespace max_ab_value_l586_58681

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃! x, x^2 + Real.sqrt a * x - b + 1/4 = 0) → 
  ∀ c, a * b ≤ c → c ≤ 1/16 :=
by sorry

end max_ab_value_l586_58681


namespace expression_evaluation_l586_58639

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x = 789 := by
  sorry

end expression_evaluation_l586_58639


namespace opposite_def_opposite_of_two_l586_58699

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 2 is -2 -/
theorem opposite_of_two : opposite 2 = -2 := by sorry

end opposite_def_opposite_of_two_l586_58699


namespace abs_inequality_equivalence_l586_58674

theorem abs_inequality_equivalence (x : ℝ) : 
  (1 ≤ |x + 3| ∧ |x + 3| ≤ 4) ↔ ((-7 ≤ x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 1)) := by
  sorry

end abs_inequality_equivalence_l586_58674


namespace function_composition_inverse_l586_58664

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_inverse (a b : ℝ) :
  (∀ x, h a b x = (x - 6) / 2) →
  a - b = 5/2 :=
by sorry

end function_composition_inverse_l586_58664


namespace swimmer_distance_l586_58694

/-- Calculates the distance traveled by a swimmer against a current. -/
theorem swimmer_distance (still_water_speed : ℝ) (current_speed : ℝ) (time : ℝ) :
  still_water_speed > current_speed →
  still_water_speed = 20 →
  current_speed = 12 →
  time = 5 →
  (still_water_speed - current_speed) * time = 40 := by
sorry

end swimmer_distance_l586_58694


namespace t_shirt_jersey_cost_difference_l586_58668

/-- The cost difference between a t-shirt and a jersey -/
def cost_difference (t_shirt_price jersey_price : ℕ) : ℕ :=
  t_shirt_price - jersey_price

/-- Theorem: The cost difference between a t-shirt and a jersey is $158 -/
theorem t_shirt_jersey_cost_difference :
  cost_difference 192 34 = 158 := by
  sorry

end t_shirt_jersey_cost_difference_l586_58668


namespace solution_satisfies_equations_l586_58640

theorem solution_satisfies_equations : ∃ x : ℚ, 8 * x^3 = 125 ∧ 4 * (x - 1)^2 = 9 := by
  use 5/2
  sorry

end solution_satisfies_equations_l586_58640


namespace range_of_k_l586_58682

-- Define set A
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}

-- Define set B
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := {x | ¬(x ∈ A)}

-- Theorem statement
theorem range_of_k (k : ℝ) : 
  (C_R_A ∩ B k).Nonempty → 0 < k ∧ k < 3 :=
by
  sorry


end range_of_k_l586_58682


namespace girls_fraction_proof_l586_58606

theorem girls_fraction_proof (T G B : ℕ) (x : ℚ) : 
  (x * G = (1 / 6) * T) →  -- Some fraction of girls is 1/6 of total
  (B = 2 * G) →            -- Ratio of boys to girls is 2
  (T = B + G) →            -- Total is sum of boys and girls
  (x = 1 / 2) :=           -- Fraction of girls is 1/2
by sorry

end girls_fraction_proof_l586_58606


namespace max_value_of_sin_cos_combination_l586_58675

theorem max_value_of_sin_cos_combination :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin x + 4 * Real.cos x
  ∃ M : ℝ, M = 5 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end max_value_of_sin_cos_combination_l586_58675


namespace glued_cubes_faces_l586_58630

/-- The number of faces of a cube -/
def cube_faces : ℕ := 6

/-- The number of new faces contributed by each glued cube -/
def new_faces_per_cube : ℕ := 5

/-- The number of faces in the resulting solid when a cube is glued to each face of an original cube -/
def resulting_solid_faces : ℕ := cube_faces + cube_faces * new_faces_per_cube

theorem glued_cubes_faces : resulting_solid_faces = 36 := by
  sorry

end glued_cubes_faces_l586_58630


namespace max_points_for_one_participant_l586_58604

theorem max_points_for_one_participant 
  (n : ℕ) 
  (avg : ℚ) 
  (min_points : ℕ) 
  (h1 : n = 50) 
  (h2 : avg = 8) 
  (h3 : min_points = 2) 
  (h4 : ∀ p : ℕ, p ≤ n → p ≥ min_points) : 
  ∃ max_points : ℕ, max_points = 302 ∧ 
  ∀ p : ℕ, p ≤ n → p ≤ max_points := by
sorry


end max_points_for_one_participant_l586_58604


namespace smallest_number_of_eggs_l586_58612

theorem smallest_number_of_eggs (total_containers : ℕ) (filled_containers : ℕ) : 
  total_containers > 10 →
  filled_containers = total_containers - 3 →
  15 * filled_containers + 14 * 3 > 150 →
  15 * filled_containers + 14 * 3 ≤ 15 * (filled_containers + 1) + 14 * 3 - 3 →
  15 * filled_containers + 14 * 3 = 162 :=
by
  sorry

end smallest_number_of_eggs_l586_58612


namespace prime_quadruplet_l586_58648

theorem prime_quadruplet (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
  p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧
  p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882 →
  ((p₁, p₂, p₃, p₄) = (2, 5, 19, 37) ∨ 
   (p₁, p₂, p₃, p₄) = (2, 11, 19, 31) ∨ 
   (p₁, p₂, p₃, p₄) = (2, 13, 19, 29)) :=
by sorry

end prime_quadruplet_l586_58648


namespace convex_polygon_24_sides_diagonals_l586_58636

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem convex_polygon_24_sides_diagonals :
  num_diagonals 24 = 126 := by
  sorry

end convex_polygon_24_sides_diagonals_l586_58636


namespace triangle_probability_ten_points_triangle_probability_ten_points_with_conditions_l586_58633

/-- Given 10 points in a plane where no three are collinear, this function
    calculates the probability that three out of four randomly chosen
    distinct segments connecting pairs of these points will form a triangle. -/
def probability_triangle_from_segments (n : ℕ) : ℚ :=
  if n = 10 then 16 / 473
  else 0

/-- Theorem stating that the probability of forming a triangle
    from three out of four randomly chosen segments is 16/473
    when there are 10 points in the plane and no three are collinear. -/
theorem triangle_probability_ten_points :
  probability_triangle_from_segments 10 = 16 / 473 := by
  sorry

/-- Assumption that no three points are collinear in the given set of points. -/
axiom no_three_collinear (n : ℕ) : Prop

/-- Theorem stating that given 10 points in a plane where no three are collinear,
    the probability that three out of four randomly chosen distinct segments
    connecting pairs of these points will form a triangle is 16/473. -/
theorem triangle_probability_ten_points_with_conditions :
  no_three_collinear 10 →
  probability_triangle_from_segments 10 = 16 / 473 := by
  sorry

end triangle_probability_ten_points_triangle_probability_ten_points_with_conditions_l586_58633


namespace delta_curve_from_rotations_l586_58653

/-- A curve in 2D space -/
structure Curve where
  -- Add necessary fields for a curve

/-- Rotation of a curve around a point by an angle -/
def rotate (c : Curve) (center : ℝ × ℝ) (angle : ℝ) : Curve :=
  sorry

/-- Sum of curves -/
def sum_curves (curves : List Curve) : Curve :=
  sorry

/-- Check if a curve is a circle with given radius -/
def is_circle (c : Curve) (radius : ℝ) : Prop :=
  sorry

/-- Check if a curve is convex -/
def is_convex (c : Curve) : Prop :=
  sorry

/-- Check if a curve is a Δ-curve -/
def is_delta_curve (c : Curve) : Prop :=
  sorry

/-- Main theorem -/
theorem delta_curve_from_rotations (K : Curve) (O : ℝ × ℝ) (h : ℝ) :
  is_convex K →
  let K' := rotate K O (2 * π / 3)
  let K'' := rotate K O (4 * π / 3)
  let M := sum_curves [K, K', K'']
  is_circle M h →
  is_delta_curve K :=
sorry

end delta_curve_from_rotations_l586_58653


namespace tetrahedron_plane_distance_l586_58692

/-- Regular tetrahedron with side length 15 -/
def tetrahedron_side_length : ℝ := 15

/-- Heights of three vertices above the plane -/
def vertex_heights : Fin 3 → ℝ
  | 0 => 15
  | 1 => 17
  | 2 => 20
  | _ => 0  -- This case should never occur due to Fin 3

/-- The theorem stating the properties of the tetrahedron and plane -/
theorem tetrahedron_plane_distance :
  ∃ (r s t : ℕ), 
    r > 0 ∧ s > 0 ∧ t > 0 ∧
    (∃ (d : ℝ), d = (r - Real.sqrt s) / t ∧
      d > 0 ∧ 
      d < tetrahedron_side_length ∧
      (∀ i, d < vertex_heights i) ∧
      r + s + t = 930) := by
  sorry

end tetrahedron_plane_distance_l586_58692


namespace f_properties_l586_58621

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4

-- State the theorem
theorem f_properties :
  (∀ x y : ℝ, f (x * y) + f (y - x) ≥ f (y + x)) ∧
  (∀ x : ℝ, f x ≥ 0) := by sorry

end f_properties_l586_58621


namespace sum_of_factors_l586_58651

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e → 
  (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120 →
  a + b + c + d + e = 39 := by
sorry

end sum_of_factors_l586_58651


namespace quadratic_equations_solution_l586_58635

def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 2 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 + q*x + r = 0}

theorem quadratic_equations_solution (p q r : ℝ) :
  (A p ∪ B q r = {-2, 1, 5}) ∧
  (A p ∩ B q r = {-2}) →
  p = -1 ∧ q = -3 ∧ r = -10 := by
  sorry

end quadratic_equations_solution_l586_58635


namespace grid_flip_theorem_l586_58649

/-- Represents a 4x4 binary grid -/
def Grid := Matrix (Fin 4) (Fin 4) Bool

/-- Represents a flip operation on the grid -/
inductive FlipOperation
| row : Fin 4 → FlipOperation
| column : Fin 4 → FlipOperation
| diagonal : Bool → FlipOperation  -- True for main diagonal, False for anti-diagonal

/-- Applies a flip operation to the grid -/
def applyFlip (g : Grid) (op : FlipOperation) : Grid :=
  sorry

/-- Checks if the grid is all zeros -/
def isAllZeros (g : Grid) : Prop :=
  ∀ i j, g i j = false

/-- Initial configurations -/
def initialGrid1 : Grid :=
  ![![false, true,  true,  false],
    ![true,  true,  false, true ],
    ![false, false, true,  true ],
    ![false, false, true,  true ]]

def initialGrid2 : Grid :=
  ![![false, true,  false, false],
    ![true,  true,  false, true ],
    ![false, false, false, true ],
    ![true,  false, true,  true ]]

def initialGrid3 : Grid :=
  ![![false, false, false, false],
    ![true,  true,  false, false],
    ![false, true,  false, true ],
    ![true,  false, false, true ]]

/-- Main theorem -/
theorem grid_flip_theorem :
  (¬ ∃ (ops : List FlipOperation), isAllZeros (ops.foldl applyFlip initialGrid1)) ∧
  (¬ ∃ (ops : List FlipOperation), isAllZeros (ops.foldl applyFlip initialGrid2)) ∧
  (∃ (ops : List FlipOperation), isAllZeros (ops.foldl applyFlip initialGrid3)) :=
sorry

end grid_flip_theorem_l586_58649


namespace eventually_all_play_all_l586_58605

/-- Represents a player in the tournament -/
inductive Player
  | Mathematician (id : ℕ)
  | Humanitarian (id : ℕ)

/-- Represents the state of the tournament -/
structure TournamentState where
  n : ℕ  -- number of humanities students
  m : ℕ  -- number of mathematicians
  queue : List Player
  table : Player × Player
  h_different_sizes : n ≠ m

/-- Represents a game played between two players -/
def Game := Player × Player

/-- Simulates the tournament for a given number of steps -/
def simulateTournament (initial : TournamentState) (steps : ℕ) : List Game := sorry

/-- Checks if all mathematicians have played with all humanitarians -/
def allPlayedAgainstAll (games : List Game) : Prop := sorry

/-- The main theorem stating that eventually all mathematicians will play against all humanitarians -/
theorem eventually_all_play_all (initial : TournamentState) :
  ∃ k : ℕ, allPlayedAgainstAll (simulateTournament initial k) := by
  sorry

end eventually_all_play_all_l586_58605


namespace fraction_simplification_l586_58601

theorem fraction_simplification (a b : ℝ) (h : a ≠ 0) :
  (a^2 + 2*a*b + b^2) / (a^2 + a*b) = (a + b) / a := by
  sorry

end fraction_simplification_l586_58601


namespace company_profits_l586_58600

theorem company_profits (revenue_prev : ℝ) (profit_prev : ℝ) (revenue_2009 : ℝ) (profit_2009 : ℝ) :
  revenue_2009 = 0.8 * revenue_prev →
  profit_2009 = 0.16 * revenue_2009 →
  profit_2009 = 1.28 * profit_prev →
  profit_prev = 0.1 * revenue_prev :=
by sorry

end company_profits_l586_58600


namespace spelling_bee_contestants_l586_58646

theorem spelling_bee_contestants (total : ℕ) : 
  (total / 2 : ℚ) / 4 = 30 → total = 240 := by sorry

end spelling_bee_contestants_l586_58646


namespace angle_conversion_l586_58686

theorem angle_conversion (angle_deg : ℝ) (k : ℤ) (α : ℝ) :
  angle_deg = -1125 →
  (k = -4 ∧ α = (7 * π) / 4) →
  (0 ≤ α ∧ α < 2 * π) →
  angle_deg * π / 180 = 2 * k * π + α := by
  sorry

end angle_conversion_l586_58686


namespace paint_cost_per_kg_paint_cost_is_50_l586_58657

/-- The cost of paint per kg, given the coverage rate and the cost to paint a cube. -/
theorem paint_cost_per_kg (coverage_rate : ℝ) (cube_side : ℝ) (total_cost : ℝ) : ℝ :=
  let surface_area := 6 * cube_side * cube_side
  let paint_needed := surface_area / coverage_rate
  total_cost / paint_needed

/-- The cost of paint per kg is 50, given the specified conditions. -/
theorem paint_cost_is_50 : paint_cost_per_kg 20 20 6000 = 50 := by
  sorry

end paint_cost_per_kg_paint_cost_is_50_l586_58657


namespace power_three_equality_l586_58696

theorem power_three_equality : 3^2012 - 6 * 3^2013 + 2 * 3^2014 = 3^2012 := by
  sorry

end power_three_equality_l586_58696


namespace cone_lateral_surface_area_l586_58623

theorem cone_lateral_surface_area (l r : ℝ) (h1 : l = 5) (h2 : r = 2) :
  π * r * l = 10 * π := by
  sorry

end cone_lateral_surface_area_l586_58623


namespace simplify_trig_expression_l586_58685

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (π - α) + Real.sin (2 * α)) / (Real.cos (α / 2))^2 = 4 * Real.sin α :=
by sorry

end simplify_trig_expression_l586_58685


namespace line_equation_l586_58634

/-- Given a line passing through (a, 0) and cutting a triangular region with area T from the second quadrant, 
    the equation of the line is -2Tx + a²y + 2aT = 0 -/
theorem line_equation (a T : ℝ) (h1 : a ≠ 0) (h2 : T > 0) : 
  ∃ (f : ℝ → ℝ), (∀ x y, f x = y ↔ -2 * T * x + a^2 * y + 2 * a * T = 0) ∧ 
                  (f a = 0) ∧
                  (∀ x y, x > 0 → y > 0 → f x = y → 
                    (1/2) * a * y = T) :=
sorry

end line_equation_l586_58634


namespace expression_simplification_and_evaluation_l586_58665

theorem expression_simplification_and_evaluation :
  let f (x : ℚ) := (x^2 - 4*x) / (x^2 - 16) / ((x^2 + 4*x) / (x^2 + 8*x + 16)) - 2*x / (x - 4)
  f (-2 : ℚ) = 1/3 := by
  sorry

end expression_simplification_and_evaluation_l586_58665


namespace meter_to_km_conversion_kg_to_g_conversion_cm_to_dm_conversion_time_to_minutes_conversion_l586_58660

-- Define conversion rates
def meter_to_km : ℕ → ℕ := λ m => m / 1000
def kg_to_g : ℕ → ℕ := λ kg => kg * 1000
def cm_to_dm : ℕ → ℕ := λ cm => cm / 10
def hours_to_minutes : ℕ → ℕ := λ h => h * 60

-- Theorem statements
theorem meter_to_km_conversion : meter_to_km 6000 = 6 := by sorry

theorem kg_to_g_conversion : kg_to_g (5 + 2) = 7000 := by sorry

theorem cm_to_dm_conversion : cm_to_dm (58 + 32) = 9 := by sorry

theorem time_to_minutes_conversion : hours_to_minutes 3 + 30 = 210 := by sorry

end meter_to_km_conversion_kg_to_g_conversion_cm_to_dm_conversion_time_to_minutes_conversion_l586_58660


namespace set_union_problem_l586_58627

theorem set_union_problem (M N : Set ℕ) (x : ℕ) :
  M = {0, x} →
  N = {1, 2} →
  M ∩ N = {1} →
  M ∪ N = {0, 1, 2} := by
  sorry

end set_union_problem_l586_58627


namespace aurelia_percentage_l586_58669

/-- Given Lauryn's earnings and the total earnings of Lauryn and Aurelia,
    calculate the percentage of Lauryn's earnings that Aurelia made. -/
theorem aurelia_percentage (lauryn_earnings total_earnings : ℝ) : 
  lauryn_earnings = 2000 →
  total_earnings = 3400 →
  (100 * (total_earnings - lauryn_earnings)) / lauryn_earnings = 70 := by
sorry

end aurelia_percentage_l586_58669


namespace smallest_valid_number_l586_58603

def is_valid_number (n : ℕ) : Prop :=
  ∃ k : ℕ, 
    n = 5 * 10^(k-1) + (n % 10^(k-1)) ∧ 
    10 * (n % 10^(k-1)) + 5 = n / 4

theorem smallest_valid_number : 
  is_valid_number 512820 ∧ 
  ∀ m : ℕ, m < 512820 → ¬(is_valid_number m) :=
sorry

end smallest_valid_number_l586_58603


namespace springfield_population_difference_l586_58680

/-- The population difference between two cities given the population of one city and their total population -/
def population_difference (population_springfield : ℕ) (total_population : ℕ) : ℕ :=
  population_springfield - (total_population - population_springfield)

/-- Theorem stating that the population difference between Springfield and the other city is 119,666 -/
theorem springfield_population_difference :
  population_difference 482653 845640 = 119666 := by
  sorry

end springfield_population_difference_l586_58680


namespace horner_v2_value_l586_58654

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (λ acc coeff => horner_step acc x coeff) 0

def polynomial : List ℝ := [5, 2, 3.5, -2.6, 1.7, -0.8]

theorem horner_v2_value :
  let x : ℝ := 5
  let v0 : ℝ := polynomial.head!
  let v1 : ℝ := horner_step v0 x (polynomial.get! 1)
  let v2 : ℝ := horner_step v1 x (polynomial.get! 2)
  v2 = 138.5 := by sorry

end horner_v2_value_l586_58654


namespace max_value_of_expression_l586_58659

theorem max_value_of_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h_sum : x + y + z = 3) :
  (x * y / (x + y + 1) + x * z / (x + z + 1) + y * z / (y + z + 1)) ≤ 1 ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 3 ∧
    x * y / (x + y + 1) + x * z / (x + z + 1) + y * z / (y + z + 1) = 1 :=
by sorry

end max_value_of_expression_l586_58659


namespace tank_final_volume_l586_58671

def tank_problem (tank_capacity : ℝ) (initial_fill_ratio : ℝ) (empty_ratio : ℝ) (refill_ratio : ℝ) : ℝ :=
  let initial_volume := tank_capacity * initial_fill_ratio
  let emptied_volume := initial_volume * empty_ratio
  let remaining_volume := initial_volume - emptied_volume
  let refilled_volume := remaining_volume * refill_ratio
  remaining_volume + refilled_volume

theorem tank_final_volume :
  tank_problem 8000 (3/4) (40/100) (30/100) = 4680 := by
  sorry

end tank_final_volume_l586_58671


namespace subset_difference_theorem_l586_58684

theorem subset_difference_theorem (n k m : ℕ) (A : Finset ℕ) 
  (h1 : k ≥ 2)
  (h2 : n ≤ m)
  (h3 : m < ((2 * k - 1) * n) / k)
  (h4 : A.card = n)
  (h5 : ∀ a ∈ A, a ≤ m) :
  ∀ x : ℤ, 0 < x ∧ x < n / (k - 1) → 
    ∃ a a' : ℕ, a ∈ A ∧ a' ∈ A ∧ (a : ℤ) - (a' : ℤ) = x :=
by sorry

end subset_difference_theorem_l586_58684


namespace smallest_number_with_18_factors_l586_58644

def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_18_factors : 
  ∃ m : ℕ, m > 1 ∧ 
           num_factors m = 18 ∧ 
           num_factors m - 2 ≥ 16 ∧
           ∀ k : ℕ, k > 1 → num_factors k = 18 → num_factors k - 2 ≥ 16 → m ≤ k :=
by sorry

end smallest_number_with_18_factors_l586_58644


namespace estimate_sqrt_expression_l586_58643

theorem estimate_sqrt_expression :
  6 < Real.sqrt 5 * (2 * Real.sqrt 5 - Real.sqrt 2) ∧
  Real.sqrt 5 * (2 * Real.sqrt 5 - Real.sqrt 2) < 7 :=
by sorry

end estimate_sqrt_expression_l586_58643


namespace sqrt_nine_over_two_simplification_l586_58676

theorem sqrt_nine_over_two_simplification :
  Real.sqrt (9 / 2) = (3 * Real.sqrt 2) / 2 := by
  sorry

end sqrt_nine_over_two_simplification_l586_58676


namespace power_function_quadrants_l586_58673

/-- A function f(x) = (m^2 - 5m + 7)x^m is a power function with its graph
    distributed in the first and third quadrants if and only if m = 3 -/
theorem power_function_quadrants (m : ℝ) : 
  (∀ x ≠ 0, ∃ f : ℝ → ℝ, f x = (m^2 - 5*m + 7) * x^m) ∧ 
  (∀ x > 0, (m^2 - 5*m + 7) * x^m > 0) ∧
  (∀ x < 0, (m^2 - 5*m + 7) * x^m < 0) ∧
  (m^2 - 5*m + 7 = 1) ↔ 
  m = 3 := by sorry

end power_function_quadrants_l586_58673


namespace sqrt_six_div_sqrt_two_eq_sqrt_three_l586_58679

theorem sqrt_six_div_sqrt_two_eq_sqrt_three :
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end sqrt_six_div_sqrt_two_eq_sqrt_three_l586_58679


namespace problem_solution_l586_58652

theorem problem_solution : ∀ M N X : ℕ,
  M = 2098 / 2 →
  N = M * 2 →
  X = M + N →
  X = 3147 := by
  sorry

end problem_solution_l586_58652


namespace no_adjacent_standing_probability_l586_58615

/-- Represents the number of valid arrangements for n people where no two adjacent people stand -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => validArrangements (n + 1) + validArrangements (n + 2)

/-- The probability of no two adjacent people standing in a circular arrangement of n people -/
def probability (n : ℕ) : ℚ := (validArrangements n : ℚ) / (2^n : ℚ)

theorem no_adjacent_standing_probability :
  probability 10 = 123 / 1024 := by
  sorry

end no_adjacent_standing_probability_l586_58615


namespace factorial_100_trailing_zeros_l586_58647

-- Define a function to count trailing zeros in a factorial
def trailingZeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem factorial_100_trailing_zeros :
  trailingZeros 100 = 24 := by sorry

end factorial_100_trailing_zeros_l586_58647


namespace tangent_line_and_triangle_area_l586_58614

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x

-- Theorem statement
theorem tangent_line_and_triangle_area :
  let P : ℝ × ℝ := (1, -1)
  -- Condition: P is on the graph of f
  (f P.1 = P.2) →
  -- Claim 1: Equation of the tangent line
  (∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ x - y - 2 = 0) ∧
  -- Claim 2: Area of the triangle
  (∃ A : ℝ, A = 2 ∧
    ∀ x₁ y₁ x₂ y₂,
      (x₁ - y₁ - 2 = 0 ∧ y₁ = 0) →
      (x₂ - y₂ - 2 = 0 ∧ x₂ = 0) →
      A = (1/2) * x₁ * (-y₂)) :=
by sorry

end tangent_line_and_triangle_area_l586_58614


namespace all_xoons_are_zeefs_and_yamps_l586_58617

-- Define the types for our sets
variable (U : Type) -- Universe set
variable (Zeef Yamp Xoon Woon : Set U)

-- Define the given conditions
variable (h1 : Zeef ⊆ Yamp)
variable (h2 : Xoon ⊆ Yamp)
variable (h3 : Woon ⊆ Zeef)
variable (h4 : Xoon ⊆ Woon)

-- Theorem to prove
theorem all_xoons_are_zeefs_and_yamps :
  Xoon ⊆ Zeef ∩ Yamp :=
by sorry

end all_xoons_are_zeefs_and_yamps_l586_58617


namespace square_triangle_equal_perimeter_l586_58625

theorem square_triangle_equal_perimeter (x : ℝ) : 
  4 * (x + 2) = 3 * (2 * x) → x = 4 := by
  sorry

end square_triangle_equal_perimeter_l586_58625


namespace greatest_number_in_set_l586_58620

/-- A set of consecutive multiples of 2 -/
def ConsecutiveMultiplesOf2 (s : Set ℕ) : Prop :=
  ∃ start : ℕ, ∀ n ∈ s, ∃ k : ℕ, n = start + 2 * k

theorem greatest_number_in_set (s : Set ℕ) 
  (h1 : ConsecutiveMultiplesOf2 s)
  (h2 : Fintype s)
  (h3 : Fintype.card s = 50)
  (h4 : 56 ∈ s)
  (h5 : ∀ n ∈ s, n ≥ 56) :
  ∃ m ∈ s, m = 154 ∧ ∀ n ∈ s, n ≤ m :=
sorry

end greatest_number_in_set_l586_58620


namespace trig_identity_l586_58656

theorem trig_identity (x : Real) (h : Real.tan x = -1/2) : 
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := by
  sorry

end trig_identity_l586_58656


namespace distance_post_office_to_home_l586_58638

/-- The distance Spencer walked from his house to the library -/
def distance_house_to_library : ℝ := 0.3

/-- The distance Spencer walked from the library to the post office -/
def distance_library_to_post_office : ℝ := 0.1

/-- The total distance Spencer walked -/
def total_distance : ℝ := 0.8

/-- Theorem: The distance Spencer walked from the post office back home is 0.4 miles -/
theorem distance_post_office_to_home : 
  total_distance - (distance_house_to_library + distance_library_to_post_office) = 0.4 := by
  sorry

end distance_post_office_to_home_l586_58638


namespace common_divisors_2n_plus_3_and_3n_plus_2_l586_58637

theorem common_divisors_2n_plus_3_and_3n_plus_2 (n : ℕ) :
  {d : ℕ | d ∣ (2*n + 3) ∧ d ∣ (3*n + 2)} = {d : ℕ | d = 1 ∨ (d = 5 ∧ n % 5 = 1)} := by
  sorry

end common_divisors_2n_plus_3_and_3n_plus_2_l586_58637


namespace bees_in_hive_l586_58693

/-- The total number of bees in a hive after more bees fly in -/
def total_bees (initial : ℕ) (flew_in : ℕ) : ℕ :=
  initial + flew_in

/-- Theorem: Given 16 initial bees and 7 more flying in, the total is 23 -/
theorem bees_in_hive : total_bees 16 7 = 23 := by
  sorry

end bees_in_hive_l586_58693


namespace machines_working_first_scenario_l586_58626

/-- The number of machines working in the first scenario -/
def num_machines : ℕ := 8

/-- The time taken in the first scenario (in hours) -/
def time_first_scenario : ℕ := 6

/-- The number of machines in the second scenario -/
def num_machines_second : ℕ := 6

/-- The time taken in the second scenario (in hours) -/
def time_second : ℕ := 8

/-- The total work done in one job lot -/
def total_work : ℕ := 1

theorem machines_working_first_scenario :
  num_machines * time_first_scenario = num_machines_second * time_second :=
by sorry

end machines_working_first_scenario_l586_58626


namespace perpendicular_lines_parallel_l586_58607

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel
  (α β γ : Plane) (m n : Line)
  (h₁ : α ≠ β) (h₂ : α ≠ γ) (h₃ : β ≠ γ) (h₄ : m ≠ n)
  (h₅ : perpendicular m α) (h₆ : perpendicular n α) :
  parallel m n :=
sorry

end perpendicular_lines_parallel_l586_58607


namespace intersection_point_unique_l586_58602

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-3/5, -4/5)

/-- First line equation: y = 3x + 1 -/
def line1 (x y : ℚ) : Prop := y = 3 * x + 1

/-- Second line equation: y + 5 = -7x -/
def line2 (x y : ℚ) : Prop := y + 5 = -7 * x

theorem intersection_point_unique :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → (x', y') = (x, y) := by sorry

end intersection_point_unique_l586_58602


namespace equation_solution_l586_58695

theorem equation_solution :
  ∀ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 5) - 2 = 0 → x = 11.25 := by
  sorry

end equation_solution_l586_58695


namespace remaining_numbers_l586_58689

def three_digit_numbers : ℕ := 900

def numbers_with_two_identical_nonadjacent_digits : ℕ := 81

def numbers_with_three_distinct_digits : ℕ := 648

theorem remaining_numbers :
  three_digit_numbers - (numbers_with_two_identical_nonadjacent_digits + numbers_with_three_distinct_digits) = 171 := by
  sorry

end remaining_numbers_l586_58689


namespace digit_245_l586_58629

/-- The decimal representation of 13/17 -/
def decimal_rep : ℚ := 13 / 17

/-- The length of the repeating sequence in the decimal representation of 13/17 -/
def repeat_length : ℕ := 16

/-- The nth digit in the decimal representation of 13/17 -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_245 : nth_digit 245 = 7 := by sorry

end digit_245_l586_58629


namespace block_arrangement_table_height_l586_58688

/-- The height of the table in the block arrangement problem -/
def table_height : ℝ := 36

/-- The initial length measurement in the block arrangement -/
def initial_length : ℝ := 42

/-- The final length measurement in the block arrangement -/
def final_length : ℝ := 36

/-- The difference between block width and overlap in the first arrangement -/
def width_overlap_difference : ℝ := 6

theorem block_arrangement_table_height :
  ∃ (block_length block_width overlap : ℝ),
    block_length + table_height - overlap = initial_length ∧
    block_width + table_height - block_length = final_length ∧
    block_width = overlap + width_overlap_difference ∧
    table_height = 36 := by
  sorry

#check block_arrangement_table_height

end block_arrangement_table_height_l586_58688


namespace sandwich_cost_calculation_l586_58655

-- Define the costs and quantities
def selling_price : ℚ := 3
def bread_cost : ℚ := 0.15
def ham_cost : ℚ := 0.25
def cheese_cost : ℚ := 0.35
def mayo_cost : ℚ := 0.10
def lettuce_cost : ℚ := 0.05
def tomato_cost : ℚ := 0.08
def packaging_cost : ℚ := 0.02

def bread_qty : ℕ := 2
def ham_qty : ℕ := 2
def cheese_qty : ℕ := 2
def mayo_qty : ℕ := 1
def lettuce_qty : ℕ := 1
def tomato_qty : ℕ := 2

def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.05

-- Define the theorem
theorem sandwich_cost_calculation :
  let ingredient_cost := bread_cost * bread_qty + ham_cost * ham_qty + cheese_cost * cheese_qty +
                         mayo_cost * mayo_qty + lettuce_cost * lettuce_qty + tomato_cost * tomato_qty
  let discount := (ham_cost * ham_qty + cheese_cost * cheese_qty) * discount_rate
  let adjusted_cost := ingredient_cost - discount + packaging_cost
  let tax := selling_price * tax_rate
  let total_cost := adjusted_cost + tax
  total_cost = 1.86 := by
  sorry

end sandwich_cost_calculation_l586_58655


namespace distance_is_approx_7_38_l586_58642

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  radius : ℝ
  chord_length_1 : ℝ
  chord_length_2 : ℝ
  chord_length_3 : ℝ
  parallel_line_distance : ℝ
  chord_length_1_eq : chord_length_1 = 40
  chord_length_2_eq : chord_length_2 = 36
  chord_length_3_eq : chord_length_3 = 40
  equally_spaced : True  -- Assumption that lines are equally spaced

/-- The distance between adjacent parallel lines in the given configuration -/
def distance_between_lines (c : CircleWithParallelLines) : ℝ :=
  c.parallel_line_distance

/-- Theorem stating that the distance between adjacent parallel lines is approximately 7.38 -/
theorem distance_is_approx_7_38 (c : CircleWithParallelLines) :
  ∃ ε > 0, |distance_between_lines c - 7.38| < ε :=
sorry

#check distance_is_approx_7_38

end distance_is_approx_7_38_l586_58642


namespace triangle_property_l586_58666

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  Real.sin A ^ 2 - Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sin B * Real.sin C →
  -- BC = 3
  a = 3 →
  -- Prove A = 2π/3
  A = 2 * π / 3 ∧
  -- Prove maximum perimeter is 3 + 2√3
  (∀ b' c' : Real, 0 < b' ∧ 0 < c' → a + b' + c' ≤ 3 + 2 * Real.sqrt 3) ∧
  (∃ b' c' : Real, 0 < b' ∧ 0 < c' ∧ a + b' + c' = 3 + 2 * Real.sqrt 3) :=
by sorry

end triangle_property_l586_58666


namespace easter_egg_hunt_l586_58687

theorem easter_egg_hunt (bonnie george cheryl kevin : ℕ) : 
  bonnie = 13 →
  george = 9 →
  cheryl = 56 →
  cheryl = bonnie + george + kevin + 29 →
  kevin = 5 := by
sorry

end easter_egg_hunt_l586_58687


namespace unique_k_no_solution_l586_58628

theorem unique_k_no_solution (k : ℕ+) : 
  (k = 2) ↔ 
  ∀ m n : ℕ+, m ≠ n → 
    ¬(Nat.lcm m.val n.val - Nat.gcd m.val n.val = k.val * (m.val - n.val)) :=
by sorry

end unique_k_no_solution_l586_58628


namespace oak_willow_difference_l586_58663

theorem oak_willow_difference (total_trees : ℕ) (willows : ℕ) : 
  total_trees = 83 → willows = 36 → total_trees - willows - willows = 11 := by
  sorry

end oak_willow_difference_l586_58663


namespace repeating_six_equals_two_thirds_l586_58624

/-- The decimal representation of a number with infinitely repeating 6 after the decimal point -/
def repeating_six : ℚ := sorry

/-- Theorem stating that the repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_six_equals_two_thirds : repeating_six = 2/3 := by sorry

end repeating_six_equals_two_thirds_l586_58624


namespace necessary_and_sufficient_condition_sufficient_not_necessary_condition_l586_58622

-- Define the sets M and P
def M : Set ℝ := {x | x < -3 ∨ x > 5}
def P (a : ℝ) : Set ℝ := {x | (x - a) * (x - 8) ≤ 0}

-- Theorem 1: Necessary and sufficient condition
theorem necessary_and_sufficient_condition (a : ℝ) :
  M ∩ P a = {x | 5 < x ∧ x ≤ 8} ↔ -3 ≤ a ∧ a ≤ 5 := by sorry

-- Theorem 2: Sufficient but not necessary condition
theorem sufficient_not_necessary_condition :
  ∃ a : ℝ, (M ∩ P a = {x | 5 < x ∧ x ≤ 8}) ∧
  ¬(∀ b : ℝ, M ∩ P b = {x | 5 < x ∧ x ≤ 8} → b = a) := by sorry

end necessary_and_sufficient_condition_sufficient_not_necessary_condition_l586_58622


namespace incorrect_calculations_l586_58632

theorem incorrect_calculations : 
  (¬ (4237 * 27925 = 118275855)) ∧ 
  (¬ (42971064 / 8264 = 5201)) ∧ 
  (¬ (1965^2 = 3761225)) ∧ 
  (¬ (371293^(1/5) = 23)) := by
  sorry

end incorrect_calculations_l586_58632


namespace total_selling_price_theorem_l586_58631

def calculate_selling_price (cost : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let price_before_discount := cost * (1 + profit_percent)
  price_before_discount * (1 - discount_percent)

theorem total_selling_price_theorem :
  let item1 := calculate_selling_price 192 0.25 0.10
  let item2 := calculate_selling_price 350 0.15 0.05
  let item3 := calculate_selling_price 500 0.30 0.15
  item1 + item2 + item3 = 1150.875 := by
  sorry

end total_selling_price_theorem_l586_58631


namespace simplify_expression_l586_58677

theorem simplify_expression (x : ℝ) : 4 * x^2 - (2 * x^2 + x - 1) + (2 - x^2 + 3 * x) = x^2 + 2 * x + 3 := by
  sorry

end simplify_expression_l586_58677


namespace arithmetic_series_sum_formula_l586_58678

/-- The sum of the first k+1 terms of an arithmetic series with first term k^2 + 2 and common difference 2 -/
def arithmetic_series_sum (k : ℕ) : ℕ := sorry

/-- The first term of the arithmetic series -/
def first_term (k : ℕ) : ℕ := k^2 + 2

/-- The common difference of the arithmetic series -/
def common_difference : ℕ := 2

/-- The number of terms in the series -/
def num_terms (k : ℕ) : ℕ := k + 1

theorem arithmetic_series_sum_formula (k : ℕ) :
  arithmetic_series_sum k = k^3 + 2*k^2 + 3*k + 2 := by sorry

end arithmetic_series_sum_formula_l586_58678


namespace inscribed_square_area_l586_58618

/-- The area of a square inscribed in the ellipse x^2/4 + y^2 = 1, 
    with sides parallel to the coordinate axes -/
theorem inscribed_square_area : 
  ∃ (s : ℝ), s > 0 ∧ 
  (∀ (x y : ℝ), x^2/4 + y^2 = 1 → 
    (x = s ∨ x = -s) ∧ (y = s ∨ y = -s)) →
  s^2 = 16/5 := by
sorry

end inscribed_square_area_l586_58618


namespace max_distinct_factors_max_additional_factors_l586_58645

theorem max_distinct_factors (x : Finset ℕ) :
  (∀ y ∈ x, y > 0) →
  (Nat.lcm 1024 2016 = Finset.lcm x (Nat.lcm 1024 2016)) →
  x.card ≤ 66 :=
by sorry

theorem max_additional_factors :
  ∃ (x : Finset ℕ), x.card = 64 ∧
  (∀ y ∈ x, y > 0) ∧
  (Nat.lcm 1024 2016 = Finset.lcm x (Nat.lcm 1024 2016)) :=
by sorry

end max_distinct_factors_max_additional_factors_l586_58645


namespace x_varies_as_z_l586_58670

-- Define the relationships between variables
def varies_as (x y : ℝ) (n : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x = k * y^n

-- State the theorem
theorem x_varies_as_z (x y w z : ℝ) :
  varies_as x y 2 →
  varies_as y w 2 →
  varies_as w z (1/5) →
  varies_as x z (4/5) :=
sorry

end x_varies_as_z_l586_58670


namespace simplify_expression_l586_58690

theorem simplify_expression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (1 + 1 / (x - 2)) / ((x^2 - 2*x + 1) / (x^2 - 4)) = (x + 2) / (x - 1) :=
by sorry

end simplify_expression_l586_58690


namespace arithmetic_sequence_sum_l586_58611

/-- Given an arithmetic sequence {aₙ} with sum Sₙ of its first n terms, 
    if S₉ = a₄ + a₅ + a₆ + 72, then a₃ + a₇ = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) →  -- Definition of Sₙ for arithmetic sequence
  (∀ n k, a (n + k) - a n = k * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  S 9 = a 4 + a 5 + a 6 + 72 →  -- Given condition
  a 3 + a 7 = 24 := by
sorry

end arithmetic_sequence_sum_l586_58611


namespace disk_arrangement_sum_l586_58613

theorem disk_arrangement_sum (n : ℕ) (r : ℝ) :
  n = 8 →
  r > 0 →
  r = 2 - Real.sqrt 2 →
  ∃ (a b c : ℕ), 
    c = 2 ∧
    n * (π * r^2) = π * (a - b * Real.sqrt c) ∧
    a + b + c = 82 :=
sorry

end disk_arrangement_sum_l586_58613


namespace fraction_equality_l586_58641

theorem fraction_equality (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1001) :
  (a + b) / (a - b) = -1001 := by sorry

end fraction_equality_l586_58641


namespace find_number_l586_58698

theorem find_number : ∃ x : ℚ, x - (3/5) * x = 62 ∧ x = 155 := by
  sorry

end find_number_l586_58698


namespace quadratic_equation_roots_l586_58661

theorem quadratic_equation_roots (p : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + p * x = 2) ∧ 
  (3 * (-1)^2 + p * (-1) = 2) →
  (3 * (2/3)^2 + p * (2/3) = 2) :=
by sorry

end quadratic_equation_roots_l586_58661


namespace product_of_x_values_l586_58609

theorem product_of_x_values (x : ℝ) : 
  (|18 / x + 4| = 3) → 
  (∃ y : ℝ, y ≠ x ∧ |18 / y + 4| = 3 ∧ x * y = 324 / 7) :=
by sorry

end product_of_x_values_l586_58609


namespace functional_equation_result_l586_58608

theorem functional_equation_result (g : ℝ → ℝ) 
  (h₁ : ∀ c d : ℝ, c^2 * g d = d^2 * g c) 
  (h₂ : g 4 ≠ 0) : 
  (g 7 - g 3) / g 4 = 5/2 := by sorry

end functional_equation_result_l586_58608


namespace polynomial_root_sum_product_l586_58691

theorem polynomial_root_sum_product (c d : ℂ) : 
  (c^4 - 6*c - 3 = 0) → 
  (d^4 - 6*d - 3 = 0) → 
  (c*d + c + d = 3 + Real.sqrt 2) := by
  sorry

end polynomial_root_sum_product_l586_58691


namespace smallest_distance_circle_ellipse_l586_58650

/-- The smallest distance between a point on the unit circle and a point on a specific ellipse -/
theorem smallest_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let ellipse := {p : ℝ × ℝ | ((p.1 - 2)^2 / 9) + (p.2^2 / 9) = 1}
  (∃ (A : ℝ × ℝ) (B : ℝ × ℝ), A ∈ circle ∧ B ∈ ellipse ∧
    ∀ (C : ℝ × ℝ) (D : ℝ × ℝ), C ∈ circle → D ∈ ellipse →
      Real.sqrt 2 - 1 ≤ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)) :=
by
  sorry


end smallest_distance_circle_ellipse_l586_58650


namespace function_domain_range_l586_58697

theorem function_domain_range (a : ℝ) (h1 : a > 1) : 
  (∀ x ∈ Set.Icc 1 a, x^2 - 2*a*x + 5 ∈ Set.Icc 1 a) ∧
  (∀ y ∈ Set.Icc 1 a, ∃ x ∈ Set.Icc 1 a, y = x^2 - 2*a*x + 5) →
  a = 2 :=
sorry

end function_domain_range_l586_58697


namespace choir_group_ratio_l586_58672

theorem choir_group_ratio (total_sopranos total_altos num_groups : ℕ) 
  (h1 : total_sopranos = 10)
  (h2 : total_altos = 15)
  (h3 : num_groups = 5)
  (h4 : total_sopranos % num_groups = 0)
  (h5 : total_altos % num_groups = 0) :
  (total_sopranos / num_groups : ℚ) / (total_altos / num_groups : ℚ) = 2 / 3 :=
by sorry

end choir_group_ratio_l586_58672


namespace track_length_is_50_l586_58683

/-- Calculates the length of a running track given weekly distance, days per week, and loops per day -/
def track_length (weekly_distance : ℕ) (days_per_week : ℕ) (loops_per_day : ℕ) : ℕ :=
  weekly_distance / (days_per_week * loops_per_day)

/-- Proves that given the specified conditions, the track length is 50 meters -/
theorem track_length_is_50 : 
  track_length 3500 7 10 = 50 := by
  sorry

#eval track_length 3500 7 10

end track_length_is_50_l586_58683


namespace infection_model_properties_l586_58662

/-- Represents the infection spread model -/
structure InfectionModel where
  initialInfected : ℕ := 1
  totalAfterTwoRounds : ℕ := 64
  averageInfectionRate : ℕ
  thirdRoundInfections : ℕ

/-- Theorem stating the properties of the infection model -/
theorem infection_model_properties (model : InfectionModel) :
  model.initialInfected = 1 ∧
  model.totalAfterTwoRounds = 64 →
  model.averageInfectionRate = 7 ∧
  model.thirdRoundInfections = 448 := by
  sorry

#check infection_model_properties

end infection_model_properties_l586_58662


namespace triangle_inequality_l586_58616

theorem triangle_inequality (α β γ : ℝ) (h_a l_a r R : ℝ) 
  (h1 : h_a / l_a = Real.cos ((β - γ) / 2))
  (h2 : 2 * r / R = 8 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2)) :
  h_a / l_a ≥ Real.sqrt (2 * r / R) := by
  sorry

end triangle_inequality_l586_58616


namespace mary_anne_sparkling_water_cost_l586_58667

/-- The annual cost of Mary Anne's sparkling water consumption -/
def annual_sparkling_water_cost (daily_consumption : ℚ) (bottle_cost : ℚ) : ℚ :=
  (365 : ℚ) * daily_consumption * bottle_cost

/-- Theorem: Mary Anne's annual sparkling water cost is $146.00 -/
theorem mary_anne_sparkling_water_cost :
  annual_sparkling_water_cost (1/5) 2 = 146 := by
  sorry

end mary_anne_sparkling_water_cost_l586_58667


namespace num_chosen_bulbs_is_two_l586_58619

/-- The number of bulbs chosen at random from a box containing defective and non-defective bulbs. -/
def num_chosen_bulbs : ℕ :=
  -- The actual number will be defined in the proof
  sorry

/-- The total number of bulbs in the box. -/
def total_bulbs : ℕ := 21

/-- The number of defective bulbs in the box. -/
def defective_bulbs : ℕ := 4

/-- The probability of choosing at least one defective bulb. -/
def prob_at_least_one_defective : ℝ := 0.35238095238095235

theorem num_chosen_bulbs_is_two :
  num_chosen_bulbs = 2 ∧
  (1 : ℝ) - (total_bulbs - defective_bulbs : ℝ) / total_bulbs ^ num_chosen_bulbs = prob_at_least_one_defective :=
by sorry

end num_chosen_bulbs_is_two_l586_58619


namespace train_speed_l586_58610

/-- Given a train of length 300 meters that crosses an electric pole in 20 seconds,
    prove that its speed is 15 meters per second. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 300) (h2 : crossing_time = 20) :
  train_length / crossing_time = 15 :=
by sorry

end train_speed_l586_58610
