import Mathlib

namespace polynomial_divisibility_and_divisor_l2076_207651

theorem polynomial_divisibility_and_divisor (m : ℤ) : 
  (∀ x : ℝ, (5 * x^2 - 9 * x + m) % (x - 2) = 0) →
  (m = -2 ∧ 2 % |m| = 0) := by
  sorry

end polynomial_divisibility_and_divisor_l2076_207651


namespace min_value_of_function_l2076_207620

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  let y := 2*x + 4/(x-1) - 1
  ∀ z, z = 2*x + 4/(x-1) - 1 → y ≤ z :=
by sorry

end min_value_of_function_l2076_207620


namespace fallen_striped_tiles_count_l2076_207681

/-- Represents the type of a tile -/
inductive TileType
| Striped
| Plain

/-- Represents the state of a tile position -/
inductive TileState
| Present
| Fallen

/-- Represents the initial checkerboard pattern -/
def initialPattern : List (List TileType) :=
  List.replicate 7 (List.replicate 7 TileType.Striped)

/-- Represents the current state of the wall after some tiles have fallen -/
def currentState : List (List TileState) :=
  [
    [TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present],
    [TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Fallen, TileState.Fallen],
    [TileState.Present, TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Fallen, TileState.Fallen, TileState.Fallen],
    [TileState.Fallen, TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Fallen, TileState.Fallen, TileState.Fallen],
    [TileState.Fallen, TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Fallen, TileState.Present, TileState.Present],
    [TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present],
    [TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present]
  ]

/-- Counts the number of fallen striped tiles -/
def countFallenStripedTiles (initial : List (List TileType)) (current : List (List TileState)) : Nat :=
  sorry

/-- Theorem: The number of fallen striped tiles is 15 -/
theorem fallen_striped_tiles_count :
  countFallenStripedTiles initialPattern currentState = 15 := by
  sorry

end fallen_striped_tiles_count_l2076_207681


namespace direct_proportion_n_value_l2076_207612

/-- A direct proportion function passing through (n, -9) with decreasing y as x increases -/
def DirectProportionFunction (n : ℝ) : ℝ → ℝ := fun x ↦ -n * x

theorem direct_proportion_n_value (n : ℝ) :
  (DirectProportionFunction n n = -9) ∧  -- The graph passes through (n, -9)
  (∀ x₁ x₂, x₁ < x₂ → DirectProportionFunction n x₁ > DirectProportionFunction n x₂) →  -- y decreases as x increases
  n = 3 := by
sorry

end direct_proportion_n_value_l2076_207612


namespace t_less_than_p_l2076_207676

theorem t_less_than_p (j p t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.8 * t) (h3 : t = 6.25) :
  (p - t) / p = 0.8 := by sorry

end t_less_than_p_l2076_207676


namespace johns_order_cost_l2076_207625

/-- The total cost of John's food order for a massive restaurant. -/
def total_cost (beef_amount : ℕ) (beef_price : ℕ) (chicken_amount_multiplier : ℕ) (chicken_price : ℕ) : ℕ :=
  beef_amount * beef_price + (beef_amount * chicken_amount_multiplier) * chicken_price

/-- Proof that John's total food order cost is $14000. -/
theorem johns_order_cost :
  total_cost 1000 8 2 3 = 14000 :=
by sorry

end johns_order_cost_l2076_207625


namespace extreme_point_property_and_max_value_l2076_207668

open Real

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2)^3 - a * x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := |f a x|

-- Theorem statement
theorem extreme_point_property_and_max_value (a : ℝ) :
  a > 0 →
  ∃ x₀ x₁ : ℝ,
    x₀ ≠ x₁ ∧
    (∀ x : ℝ, f a x₀ ≥ f a x ∨ f a x₀ ≤ f a x) ∧
    f a x₀ = f a x₁ →
    x₁ + 2 * x₀ = 6 ∧
    (∀ x : ℝ, x ∈ Set.Icc 0 6 → g a x ≤ 40) ∧
    (∃ x : ℝ, x ∈ Set.Icc 0 6 ∧ g a x = 40) →
    a = 4 ∨ a = 12 :=
by sorry

end extreme_point_property_and_max_value_l2076_207668


namespace expression_evaluation_l2076_207667

theorem expression_evaluation (a : ℝ) (h : a = 2023) : 
  ((a + 1) / (a - 1) + 1) / (2 * a / (a^2 - 1)) = 2024 := by
  sorry

end expression_evaluation_l2076_207667


namespace sum_of_three_consecutive_cubes_divisible_by_nine_l2076_207673

theorem sum_of_three_consecutive_cubes_divisible_by_nine (a : ℤ) :
  ∃ k : ℤ, a^3 + (a+1)^3 + (a+2)^3 = 9 * k := by
  sorry

end sum_of_three_consecutive_cubes_divisible_by_nine_l2076_207673


namespace regular_polygon_sides_l2076_207604

theorem regular_polygon_sides (exterior_angle : ℝ) : 
  exterior_angle = 45 → (360 : ℝ) / exterior_angle = 8 := by
  sorry

end regular_polygon_sides_l2076_207604


namespace scarves_per_box_l2076_207677

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_clothing : ℕ) : 
  num_boxes = 8 → 
  mittens_per_box = 6 → 
  total_clothing = 80 → 
  (total_clothing - num_boxes * mittens_per_box) / num_boxes = 4 := by
sorry

end scarves_per_box_l2076_207677


namespace power_product_cube_l2076_207686

theorem power_product_cube (R : Type*) [CommRing R] (x y : R) :
  (x * y^2)^3 = x^3 * y^6 := by sorry

end power_product_cube_l2076_207686


namespace inverse_sum_bound_l2076_207635

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

-- State the theorem
theorem inverse_sum_bound 
  (k : ℝ) (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < 2)
  (h4 : f k α = 0) (h5 : f k β = 0) :
  1/α + 1/β < 4 :=
by sorry

end inverse_sum_bound_l2076_207635


namespace league_teams_count_l2076_207617

theorem league_teams_count (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end league_teams_count_l2076_207617


namespace total_amount_paid_l2076_207652

def apple_quantity : ℕ := 8
def apple_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 45

theorem total_amount_paid : 
  apple_quantity * apple_rate + mango_quantity * mango_rate = 965 := by
  sorry

end total_amount_paid_l2076_207652


namespace range_of_a_l2076_207627

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4*x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 := by sorry

end range_of_a_l2076_207627


namespace municipal_hiring_problem_l2076_207695

theorem municipal_hiring_problem (U P : Finset ℕ) 
  (h1 : U.card = 120)
  (h2 : P.card = 98)
  (h3 : (U ∩ P).card = 40) :
  (U ∪ P).card = 218 := by
sorry

end municipal_hiring_problem_l2076_207695


namespace line_plane_perpendicular_parallel_parallel_lines_perpendicular_planes_l2076_207621

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)

-- Define the given conditions
variable (l₁ l₂ : Line) (α β : Plane)
variable (h1 : perpendicular l₁ α)
variable (h2 : contains β l₂)

-- Theorem to prove
theorem line_plane_perpendicular_parallel 
  (h3 : parallel α β) : perpendicularLines l₁ l₂ :=
sorry

theorem parallel_lines_perpendicular_planes 
  (h4 : perpendicularLines l₁ l₂) : perpendicularPlanes α β :=
sorry

end line_plane_perpendicular_parallel_parallel_lines_perpendicular_planes_l2076_207621


namespace zeros_of_f_shifted_l2076_207632

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem zeros_of_f_shifted (x : ℝ) : 
  f (x - 1) = 0 ↔ x = 0 ∨ x = 2 := by
  sorry

end zeros_of_f_shifted_l2076_207632


namespace gnomon_shadow_length_l2076_207654

/-- Given a candle and a gnomon, this theorem calculates the length of the shadow cast by the gnomon. -/
theorem gnomon_shadow_length 
  (h : ℝ) -- height of the candle
  (H : ℝ) -- height of the gnomon
  (d : ℝ) -- distance between the bases of the candle and gnomon
  (h_pos : h > 0)
  (H_pos : H > 0)
  (d_pos : d > 0)
  (H_gt_h : H > h) :
  ∃ x : ℝ, x = (h * d) / (H - h) ∧ x > 0 := by
  sorry

end gnomon_shadow_length_l2076_207654


namespace average_of_three_numbers_l2076_207680

theorem average_of_three_numbers (y : ℝ) : (15 + 24 + y) / 3 = 20 → y = 21 := by
  sorry

end average_of_three_numbers_l2076_207680


namespace negative_fraction_comparison_l2076_207642

theorem negative_fraction_comparison : -4/3 < -5/4 := by
  sorry

end negative_fraction_comparison_l2076_207642


namespace xy_value_l2076_207665

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^2 + y^2 = 2) (h2 : x^4 + y^4 = 14/8) : x * y = 3 * Real.sqrt 2 / 4 := by
  sorry

end xy_value_l2076_207665


namespace digit_sum_puzzle_l2076_207694

theorem digit_sum_puzzle (a b : ℕ) : 
  a ∈ (Set.Icc 1 9) → 
  b ∈ (Set.Icc 1 9) → 
  82 * 10 * a + 7 + 6 * b = 190 → 
  a + 2 * b = 7 := by
sorry

end digit_sum_puzzle_l2076_207694


namespace positive_correlation_from_arrangement_l2076_207670

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A scatter plot is a list of points -/
def ScatterPlot := List Point

/-- 
  A function that determines if a scatter plot has a general 
  bottom-left to top-right arrangement 
-/
def isBottomLeftToTopRight (plot : ScatterPlot) : Prop :=
  sorry

/-- 
  A function that calculates the correlation coefficient 
  between x and y coordinates in a scatter plot
-/
def correlationCoefficient (plot : ScatterPlot) : ℝ :=
  sorry

/-- 
  Theorem: If a scatter plot has a general bottom-left to top-right arrangement,
  then the correlation between x and y coordinates is positive
-/
theorem positive_correlation_from_arrangement (plot : ScatterPlot) :
  isBottomLeftToTopRight plot → correlationCoefficient plot > 0 := by
  sorry

end positive_correlation_from_arrangement_l2076_207670


namespace simplify_expression_l2076_207636

theorem simplify_expression (x : ℝ) : (3 * x + 30) + (150 * x - 45) = 153 * x - 15 := by
  sorry

end simplify_expression_l2076_207636


namespace inequality_proof_l2076_207618

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end inequality_proof_l2076_207618


namespace train_stations_distance_l2076_207689

/-- The distance between two stations given train meeting points -/
theorem train_stations_distance
  (first_meet_offset : ℝ)  -- Distance from midpoint to first meeting point
  (second_meet_distance : ℝ)  -- Distance from eastern station to second meeting point
  (h1 : first_meet_offset = 10)  -- First meeting 10 km west of midpoint
  (h2 : second_meet_distance = 40)  -- Second meeting 40 km from eastern station
  : ℝ :=
by
  -- The distance between the stations
  let distance : ℝ := 140
  -- Proof goes here
  sorry

#check train_stations_distance

end train_stations_distance_l2076_207689


namespace candy_solution_l2076_207675

/-- Represents the candy distribution problem -/
def candy_problem (billy_initial caleb_initial andy_initial father_bought billy_received caleb_received : ℕ) : Prop :=
  let andy_received := father_bought - billy_received - caleb_received
  let billy_final := billy_initial + billy_received
  let caleb_final := caleb_initial + caleb_received
  let andy_final := andy_initial + andy_received
  andy_final - caleb_final = 4

/-- Theorem stating the solution to the candy distribution problem -/
theorem candy_solution :
  candy_problem 6 11 9 36 8 11 := by
  sorry

#check candy_solution

end candy_solution_l2076_207675


namespace combinations_equal_twenty_l2076_207639

/-- The number of paint colors available. -/
def num_colors : ℕ := 5

/-- The number of painting methods available. -/
def num_methods : ℕ := 4

/-- The total number of combinations of paint colors and painting methods. -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 20. -/
theorem combinations_equal_twenty : total_combinations = 20 := by
  sorry

end combinations_equal_twenty_l2076_207639


namespace counterexample_square_inequality_l2076_207658

theorem counterexample_square_inequality : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by
  sorry

end counterexample_square_inequality_l2076_207658


namespace arrange_digits_eq_16_l2076_207661

/-- The number of ways to arrange the digits of 47,770 into a 5-digit number not beginning with 0 -/
def arrange_digits : ℕ :=
  let digits : List ℕ := [4, 7, 7, 7, 0]
  let total_digits : ℕ := 5
  let non_zero_digits : ℕ := 4
  let repeated_digit : ℕ := 7
  let repeated_count : ℕ := 3

  /- Number of ways to place 0 in the last 4 positions -/
  let zero_placements : ℕ := total_digits - 1

  /- Number of ways to arrange the remaining digits -/
  let remaining_arrangements : ℕ := Nat.factorial non_zero_digits / Nat.factorial repeated_count

  zero_placements * remaining_arrangements

theorem arrange_digits_eq_16 : arrange_digits = 16 := by
  sorry

end arrange_digits_eq_16_l2076_207661


namespace candidate_A_percentage_l2076_207646

def total_votes : ℕ := 560000
def invalid_vote_percentage : ℚ := 15 / 100
def valid_votes_for_A : ℕ := 380800

theorem candidate_A_percentage :
  (valid_votes_for_A : ℚ) / ((1 - invalid_vote_percentage) * total_votes) * 100 = 80 := by
  sorry

end candidate_A_percentage_l2076_207646


namespace prime_power_plus_two_l2076_207649

theorem prime_power_plus_two (p : ℕ) : 
  Prime p → Prime (p^2 + 2) → Prime (p^3 + 2) := by
  sorry

end prime_power_plus_two_l2076_207649


namespace fraction_simplification_l2076_207659

theorem fraction_simplification (c : ℝ) : (5 + 6 * c) / 9 + 3 = (32 + 6 * c) / 9 := by
  sorry

end fraction_simplification_l2076_207659


namespace smallest_perimeter_isosceles_triangle_l2076_207628

/-- Triangle with positive integer side lengths --/
structure IsoscelesTriangle where
  pq : ℕ+
  qr : ℕ+

/-- Angle bisector intersection point --/
structure AngleBisectorIntersection where
  qj : ℝ

/-- Theorem statement for the smallest perimeter of the isosceles triangle --/
theorem smallest_perimeter_isosceles_triangle
  (t : IsoscelesTriangle)
  (j : AngleBisectorIntersection)
  (h1 : j.qj = 10) :
  2 * (t.pq + t.qr) ≥ 416 ∧
  ∃ (t' : IsoscelesTriangle), 2 * (t'.pq + t'.qr) = 416 := by
  sorry

end smallest_perimeter_isosceles_triangle_l2076_207628


namespace quadrilateral_angle_measure_l2076_207678

theorem quadrilateral_angle_measure (A B C D : ℝ) : 
  A = 105 → B = C → A + B + C + D = 360 → D = 180 := by sorry

end quadrilateral_angle_measure_l2076_207678


namespace parabola_directrix_theorem_l2076_207655

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ := sorry

/-- A parabola opens upward if a > 0 -/
def opens_upward (p : Parabola) : Prop := p.a > 0

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_directrix_theorem (p : Parabola) :
  p.a = 1/4 ∧ p.b = 0 ∧ p.c = 0 ∧ opens_upward p ∧ vertex p = (0, 0) →
  directrix p = -1/2 := by sorry

end parabola_directrix_theorem_l2076_207655


namespace total_silk_dyed_l2076_207613

theorem total_silk_dyed (green_silk : ℕ) (pink_silk : ℕ) 
  (h1 : green_silk = 61921) (h2 : pink_silk = 49500) : 
  green_silk + pink_silk = 111421 := by
  sorry

end total_silk_dyed_l2076_207613


namespace fourth_root_of_256000000_l2076_207664

theorem fourth_root_of_256000000 : (256000000 : ℝ) ^ (1/4 : ℝ) = 40 * (10 : ℝ).sqrt := by sorry

end fourth_root_of_256000000_l2076_207664


namespace octagon_area_l2076_207663

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
theorem octagon_area (r : ℝ) (h : r = 3) : 
  let s := 2 * r * Real.sqrt ((1 - 1 / Real.sqrt 2) / 2)
  let area_triangle := 1 / 2 * s^2 * (1 / Real.sqrt 2)
  8 * area_triangle = 48 * (2 - Real.sqrt 2) := by
  sorry

end octagon_area_l2076_207663


namespace x_greater_than_e_l2076_207660

theorem x_greater_than_e (x : ℝ) (h1 : Real.log x > 0) (h2 : x > 1) : x > Real.exp 1 := by
  sorry

end x_greater_than_e_l2076_207660


namespace isosceles_triangle_perimeter_l2076_207672

theorem isosceles_triangle_perimeter (base height : ℝ) (h1 : base = 10) (h2 : height = 6) :
  let side := Real.sqrt (height ^ 2 + (base / 2) ^ 2)
  2 * side + base = 2 * Real.sqrt 61 + 10 := by
sorry

end isosceles_triangle_perimeter_l2076_207672


namespace gcf_of_180_and_126_l2076_207616

theorem gcf_of_180_and_126 : Nat.gcd 180 126 = 18 := by
  sorry

end gcf_of_180_and_126_l2076_207616


namespace pyramid_surface_area_change_l2076_207666

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height

/-- Represents a quadrilateral pyramid -/
structure QuadPyramid where
  base : Point3D  -- center of the base
  apex : Point3D

/-- Calculates the surface area of a quadrilateral pyramid -/
def surfaceArea (p : Parallelepiped) (q : QuadPyramid) : ℝ := sorry

/-- Position of the apex on L₂ -/
inductive ApexPosition
  | Midpoint
  | Between
  | Vertex

/-- Theorem about the surface area of the pyramid -/
theorem pyramid_surface_area_change
  (p : Parallelepiped)
  (q : QuadPyramid)
  (h₁ : q.base.z = 0)  -- base is on the xy-plane
  (h₂ : q.apex.z = p.c)  -- apex is on the top face
  :
  (∀ (pos₁ pos₂ : ApexPosition),
    pos₁ = ApexPosition.Midpoint ∧ pos₂ = ApexPosition.Between →
      surfaceArea p q < surfaceArea p { q with apex := sorry }) ∧
  (∀ (pos₁ pos₂ : ApexPosition),
    pos₁ = ApexPosition.Between ∧ pos₂ = ApexPosition.Vertex →
      surfaceArea p q < surfaceArea p { q with apex := sorry }) ∧
  (∀ (pos : ApexPosition),
    pos = ApexPosition.Vertex →
      ∀ (q' : QuadPyramid), surfaceArea p q ≤ surfaceArea p q') :=
sorry

end pyramid_surface_area_change_l2076_207666


namespace trajectory_and_intersection_l2076_207648

-- Define the line l: x - y + a = 0
def line_l (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Define points M and N
def point_M : ℝ × ℝ := (-2, 0)
def point_N : ℝ × ℝ := (-1, 0)

-- Define the distance ratio condition for point Q
def distance_ratio (x y : ℝ) : Prop :=
  Real.sqrt ((x + 2)^2 + y^2) / Real.sqrt ((x + 1)^2 + y^2) = Real.sqrt 2

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the perpendicularity condition
def perpendicular_vectors (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_intersection :
  -- Part I: Prove that the trajectory of Q is the circle C
  (∀ x y : ℝ, distance_ratio x y ↔ circle_C x y) ∧
  -- Part II: Prove that when l intersects C at two points with perpendicular position vectors, a = ±√2
  (∀ a x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    line_l a x₁ y₁ ∧ line_l a x₂ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    perpendicular_vectors x₁ y₁ x₂ y₂ →
    a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by sorry

end trajectory_and_intersection_l2076_207648


namespace expression_factorization_l2076_207699

theorem expression_factorization (y : ℝ) :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := by
  sorry

end expression_factorization_l2076_207699


namespace milk_composition_equation_l2076_207644

/-- Represents the nutritional composition of a bottle of milk -/
structure MilkComposition where
  protein : ℝ
  fat : ℝ
  carbohydrate : ℝ

/-- The total content of carbohydrates, protein, and fat in grams -/
def total_content : ℝ := 30

/-- Theorem stating the correct equation for the milk composition -/
theorem milk_composition_equation (m : MilkComposition) 
  (h1 : m.carbohydrate = 1.5 * m.protein)
  (h2 : m.carbohydrate + m.protein + m.fat = total_content) :
  (5/2) * m.protein + m.fat = total_content := by
  sorry

end milk_composition_equation_l2076_207644


namespace cube_root_of_four_fifth_powers_l2076_207696

theorem cube_root_of_four_fifth_powers (x : ℝ) :
  x = (5^7 + 5^7 + 5^7 + 5^7)^(1/3) → x = 100 * 10^(1/3) := by
  sorry

end cube_root_of_four_fifth_powers_l2076_207696


namespace racecar_repair_discount_l2076_207638

/-- Calculates the discount percentage on a racecar repair --/
theorem racecar_repair_discount (original_cost prize keep_percentage profit : ℝ) :
  original_cost = 20000 →
  prize = 70000 →
  keep_percentage = 0.9 →
  profit = 47000 →
  (original_cost - (keep_percentage * prize - profit)) / original_cost = 0.2 := by
  sorry

end racecar_repair_discount_l2076_207638


namespace max_value_of_z_l2076_207611

theorem max_value_of_z (x y : ℝ) (h1 : |x| + |y| ≤ 4) (h2 : 2*x + y - 4 ≤ 0) :
  ∃ (z : ℝ), z = 2*x - y ∧ z ≤ 20/3 ∧ ∃ (x' y' : ℝ), |x'| + |y'| ≤ 4 ∧ 2*x' + y' - 4 ≤ 0 ∧ 2*x' - y' = 20/3 :=
sorry

end max_value_of_z_l2076_207611


namespace movies_watched_l2076_207631

theorem movies_watched (total : ℕ) (book_movie_diff : ℕ) : 
  total = 13 ∧ book_movie_diff = 1 → 
  ∃ (books movies : ℕ), books = movies + book_movie_diff ∧ 
                         books + movies = total ∧ 
                         movies = 6 := by
  sorry

end movies_watched_l2076_207631


namespace pens_per_student_l2076_207684

theorem pens_per_student (total_pens : ℕ) (total_pencils : ℕ) (num_students : ℕ) : 
  total_pens = 1001 → total_pencils = 910 → num_students = 91 →
  total_pens / num_students = 11 :=
by
  sorry

end pens_per_student_l2076_207684


namespace hcl_formed_equals_c2h6_available_l2076_207615

-- Define the chemical reaction
structure Reaction where
  c2h6 : ℝ
  cl2 : ℝ
  c2h5cl : ℝ
  hcl : ℝ

-- Define the stoichiometric coefficients
def stoichiometric_ratio : Reaction :=
  { c2h6 := 1, cl2 := 1, c2h5cl := 1, hcl := 1 }

-- Define the available moles of reactants
def available_reactants : Reaction :=
  { c2h6 := 3, cl2 := 6, c2h5cl := 0, hcl := 0 }

-- Theorem: The number of moles of HCl formed is equal to the number of moles of C2H6 available
theorem hcl_formed_equals_c2h6_available :
  available_reactants.hcl = available_reactants.c2h6 :=
by
  sorry


end hcl_formed_equals_c2h6_available_l2076_207615


namespace negation_of_existence_negation_of_proposition_l2076_207690

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬p x) := by sorry

theorem negation_of_proposition : 
  (¬∃ x₀ : ℝ, (2 : ℝ)^x₀ ≠ 1) ↔ (∀ x₀ : ℝ, (2 : ℝ)^x₀ = 1) := by sorry

end negation_of_existence_negation_of_proposition_l2076_207690


namespace tournament_rounds_l2076_207688

/-- Represents a table tennis tournament with the given rules --/
structure TableTennisTournament where
  players : ℕ
  champion_losses : ℕ

/-- Calculates the number of rounds in the tournament --/
def rounds (t : TableTennisTournament) : ℕ :=
  2 * (t.players - 1) + t.champion_losses

/-- Theorem stating that a tournament with 15 players and a champion who lost once has 29 rounds --/
theorem tournament_rounds :
  ∀ t : TableTennisTournament,
    t.players = 15 →
    t.champion_losses = 1 →
    rounds t = 29 :=
by
  sorry

#check tournament_rounds

end tournament_rounds_l2076_207688


namespace symmetric_point_coordinates_l2076_207653

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the y-axis -/
def symmetricAboutYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

/-- The theorem stating that if A(2,5) is symmetric with B about the y-axis, then B(-2,5) -/
theorem symmetric_point_coordinates :
  let A : Point := ⟨2, 5⟩
  let B : Point := ⟨-2, 5⟩
  symmetricAboutYAxis A B → B = ⟨-2, 5⟩ := by
  sorry

end symmetric_point_coordinates_l2076_207653


namespace special_sequence_14th_term_l2076_207619

/-- A sequence of positive real numbers satisfying certain conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (a 2 = 2) ∧ 
  (a 8 = 8) ∧ 
  (∀ n ≥ 2, Real.sqrt (a (n - 1)) * Real.sqrt (a (n + 1)) = a n)

/-- The 14th term of the special sequence is 32 -/
theorem special_sequence_14th_term (a : ℕ → ℝ) (h : SpecialSequence a) : a 14 = 32 := by
  sorry

end special_sequence_14th_term_l2076_207619


namespace min_bullseyes_is_52_l2076_207698

/-- The number of shots in the archery tournament -/
def total_shots : ℕ := 120

/-- Chelsea's minimum score on each shot -/
def chelsea_min_score : ℕ := 5

/-- Score for a bullseye -/
def bullseye_score : ℕ := 12

/-- Chelsea's lead at halfway point -/
def chelsea_lead : ℕ := 60

/-- The number of shots taken so far -/
def shots_taken : ℕ := total_shots / 2

/-- Function to calculate the minimum number of bullseyes Chelsea needs to guarantee victory -/
def min_bullseyes_for_victory : ℕ :=
  let max_opponent_score := shots_taken * bullseye_score + chelsea_lead
  let chelsea_non_bullseye_score := (total_shots - shots_taken) * chelsea_min_score
  ((max_opponent_score - chelsea_non_bullseye_score) / (bullseye_score - chelsea_min_score)) + 1

/-- Theorem stating that the minimum number of bullseyes Chelsea needs is 52 -/
theorem min_bullseyes_is_52 : min_bullseyes_for_victory = 52 := by
  sorry

end min_bullseyes_is_52_l2076_207698


namespace alternating_seating_card_sum_l2076_207600

theorem alternating_seating_card_sum (n : ℕ) (h : n ≥ 3) :
  ∃ (m : ℕ),
    (∀ (i : ℕ) (hi : i ≤ n), ∃ (b : ℕ), b ≤ n ∧ b = i) ∧  -- Boys' cards
    (∀ (j : ℕ) (hj : n < j ∧ j ≤ 2*n), ∃ (g : ℕ), n < g ∧ g ≤ 2*n ∧ g = j) ∧  -- Girls' cards
    (∀ (k : ℕ) (hk : k ≤ n),
      ∃ (b g₁ g₂ : ℕ),
        b ≤ n ∧ n < g₁ ∧ g₁ ≤ 2*n ∧ n < g₂ ∧ g₂ ≤ 2*n ∧
        b + g₁ + g₂ = m) ↔
  Odd n :=
by sorry

end alternating_seating_card_sum_l2076_207600


namespace K_3_15_5_l2076_207692

def K (x y z : ℚ) : ℚ := x / y + y / z + z / x

theorem K_3_15_5 : K 3 15 5 = 73 / 15 := by
  sorry

end K_3_15_5_l2076_207692


namespace full_seasons_count_l2076_207669

/-- The number of days until the final season premiere -/
def days_until_premiere : ℕ := 10

/-- The number of episodes per season -/
def episodes_per_season : ℕ := 15

/-- The number of episodes Joe watches per day -/
def episodes_per_day : ℕ := 6

/-- The number of full seasons already aired -/
def full_seasons : ℕ := (days_until_premiere * episodes_per_day) / episodes_per_season

theorem full_seasons_count : full_seasons = 4 := by
  sorry

end full_seasons_count_l2076_207669


namespace cannot_determine_charles_loss_l2076_207640

def willie_initial : ℕ := 48
def charles_initial : ℕ := 14
def willie_future : ℕ := 13

theorem cannot_determine_charles_loss :
  ∀ (charles_loss : ℕ),
  ∃ (willie_loss : ℕ),
  willie_initial - willie_loss = willie_future ∧
  charles_initial ≥ charles_loss ∧
  ∃ (charles_loss' : ℕ),
  charles_loss' ≠ charles_loss ∧
  charles_initial ≥ charles_loss' ∧
  willie_initial - willie_loss = willie_future :=
by sorry

end cannot_determine_charles_loss_l2076_207640


namespace cupcakes_sold_katie_sold_20_l2076_207691

/-- Represents the cupcake sale problem -/
def cupcake_sale (initial : ℕ) (additional : ℕ) (final : ℕ) : ℕ :=
  initial + additional - final

/-- Theorem: The number of cupcakes sold is equal to the total made minus the final number -/
theorem cupcakes_sold (initial additional final : ℕ) :
  cupcake_sale initial additional final = (initial + additional) - final :=
by
  sorry

/-- Corollary: In Katie's specific case, she sold 20 cupcakes -/
theorem katie_sold_20 :
  cupcake_sale 26 20 26 = 20 :=
by
  sorry

end cupcakes_sold_katie_sold_20_l2076_207691


namespace intersection_with_complement_l2076_207643

-- Define the sets
def U : Set ℕ := Set.univ
def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 2, 3, 6, 8}

-- State the theorem
theorem intersection_with_complement : A ∩ (U \ B) = {4, 5} := by sorry

end intersection_with_complement_l2076_207643


namespace ball_max_height_l2076_207601

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 45

/-- Theorem stating the maximum height reached by the ball -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 69.5 := by
  sorry

end ball_max_height_l2076_207601


namespace unique_acute_prime_angled_triangle_l2076_207656

-- Define a structure for triangles
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define what it means for a triangle to be acute
def isAcute (t : Triangle) : Prop :=
  t.a < 90 ∧ t.b < 90 ∧ t.c < 90

-- Define what it means for a triangle to have prime angles
def hasPrimeAngles (t : Triangle) : Prop :=
  isPrime t.a ∧ isPrime t.b ∧ isPrime t.c

-- Define what it means for a triangle to be valid (sum of angles is 180)
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b + t.c = 180

-- Theorem statement
theorem unique_acute_prime_angled_triangle :
  ∃! t : Triangle, isAcute t ∧ hasPrimeAngles t ∧ isValidTriangle t ∧
  t.a = 2 ∧ t.b = 89 ∧ t.c = 89 :=
sorry

end unique_acute_prime_angled_triangle_l2076_207656


namespace system_solution_l2076_207682

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y = 1) ∧ (y + z = 2) ∧ (z + x = 3) ∧ (x = 1) ∧ (y = 0) ∧ (z = 2) := by
  sorry

end system_solution_l2076_207682


namespace unattainable_y_value_l2076_207622

theorem unattainable_y_value (x : ℝ) (h : x ≠ -4/3) :
  ¬∃ x, (2 - x) / (3 * x + 4) = -1/3 :=
by sorry

end unattainable_y_value_l2076_207622


namespace people_on_stairs_l2076_207626

/-- The number of ways to arrange people on steps. -/
def arrange_people (num_people : ℕ) (num_steps : ℕ) (max_per_step : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem. -/
theorem people_on_stairs :
  arrange_people 4 7 3 = 2394 := by
  sorry

end people_on_stairs_l2076_207626


namespace sufficient_not_necessary_condition_l2076_207623

theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x^2 - 2*x < 0 → abs x < 2) ∧
  (∃ x : ℝ, abs x < 2 ∧ ¬(x^2 - 2*x < 0)) :=
by sorry

end sufficient_not_necessary_condition_l2076_207623


namespace mike_earnings_l2076_207671

/-- Mike's total earnings for the week -/
def total_earnings (first_job_wages : ℕ) (second_job_hours : ℕ) (second_job_rate : ℕ) : ℕ :=
  first_job_wages + second_job_hours * second_job_rate

/-- Theorem stating Mike's total earnings for the week -/
theorem mike_earnings : 
  total_earnings 52 12 9 = 160 := by
  sorry

end mike_earnings_l2076_207671


namespace abs_diff_segments_of_cyclic_quad_with_incircle_l2076_207679

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadWithIncircle where
  -- Side lengths of the quadrilateral
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Conditions for a valid quadrilateral
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  -- Condition for cyclic quadrilateral (sum of opposite sides are equal)
  cyclic : a + c = b + d
  -- Additional condition for having an inscribed circle
  has_incircle : True

/-- Theorem stating the absolute difference between segments -/
theorem abs_diff_segments_of_cyclic_quad_with_incircle 
  (q : CyclicQuadWithIncircle) 
  (h1 : q.a = 80) 
  (h2 : q.b = 100) 
  (h3 : q.c = 140) 
  (h4 : q.d = 120) 
  (x y : ℝ) 
  (h5 : x + y = q.c) : 
  |x - y| = 166.36 := by
  sorry

end abs_diff_segments_of_cyclic_quad_with_incircle_l2076_207679


namespace square_area_ratio_l2076_207647

theorem square_area_ratio (small_side : ℝ) (large_side : ℝ) 
  (h1 : small_side = 2) 
  (h2 : large_side = 5) : 
  (small_side^2) / ((large_side^2 / 2) - (small_side^2 / 2)) = 8 / 21 := by
  sorry

end square_area_ratio_l2076_207647


namespace gradient_and_magnitude_at_point_l2076_207609

/-- The function z(x, y) = 3x^2 - 2y^2 -/
def z (x y : ℝ) : ℝ := 3 * x^2 - 2 * y^2

/-- The gradient of z at point (x, y) -/
def grad_z (x y : ℝ) : ℝ × ℝ := (6 * x, -4 * y)

theorem gradient_and_magnitude_at_point :
  let p : ℝ × ℝ := (1, 2)
  (grad_z p.1 p.2 = (6, -8)) ∧
  (Real.sqrt ((grad_z p.1 p.2).1^2 + (grad_z p.1 p.2).2^2) = 10) := by
  sorry

end gradient_and_magnitude_at_point_l2076_207609


namespace complex_number_in_second_quadrant_l2076_207685

theorem complex_number_in_second_quadrant :
  let z : ℂ := (2 * Complex.I) / (2 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end complex_number_in_second_quadrant_l2076_207685


namespace probability_two_red_one_green_l2076_207633

def total_marbles : ℕ := 4 + 5 + 3 + 2

def red_marbles : ℕ := 4
def green_marbles : ℕ := 5

def marbles_drawn : ℕ := 3

theorem probability_two_red_one_green :
  (Nat.choose red_marbles 2 * Nat.choose green_marbles 1) / Nat.choose total_marbles marbles_drawn = 15 / 182 :=
sorry

end probability_two_red_one_green_l2076_207633


namespace ellen_yogurt_amount_l2076_207637

/-- The amount of yogurt used in Ellen's smoothie -/
def yogurt_amount (strawberries orange_juice total : ℝ) : ℝ :=
  total - (strawberries + orange_juice)

/-- Theorem: Ellen used 0.1 cup of yogurt in her smoothie -/
theorem ellen_yogurt_amount :
  yogurt_amount 0.2 0.2 0.5 = 0.1 := by
  sorry

end ellen_yogurt_amount_l2076_207637


namespace negative_abs_negative_eight_l2076_207697

theorem negative_abs_negative_eight : -|-8| = -8 := by
  sorry

end negative_abs_negative_eight_l2076_207697


namespace max_value_expression_l2076_207674

theorem max_value_expression :
  (∃ x : ℝ, |x - 1| - |x + 4| - 5 = 0) ∧
  (∀ x : ℝ, |x - 1| - |x + 4| - 5 ≤ 0) := by
sorry

end max_value_expression_l2076_207674


namespace special_collection_books_l2076_207606

/-- The number of books in the special collection at the beginning of the month -/
def initial_books : ℕ := 75

/-- The percentage of loaned books that are returned -/
def return_rate : ℚ := 65 / 100

/-- The number of books in the special collection at the end of the month -/
def final_books : ℕ := 54

/-- The number of books loaned out during the month -/
def loaned_books : ℚ := 60.00000000000001

theorem special_collection_books :
  initial_books = final_books + (loaned_books - loaned_books * return_rate).ceil :=
sorry

end special_collection_books_l2076_207606


namespace housing_units_with_vcr_l2076_207662

theorem housing_units_with_vcr (H : ℝ) (H_pos : H > 0) : 
  let cable_tv := (1 / 5 : ℝ) * H
  let vcr := F * H
  let both := (1 / 4 : ℝ) * cable_tv
  let neither := (3 / 4 : ℝ) * H
  ∃ F : ℝ, F = (1 / 10 : ℝ) ∧ cable_tv + vcr - both = H - neither :=
by sorry

end housing_units_with_vcr_l2076_207662


namespace badminton_cost_comparison_l2076_207629

/-- Cost calculation for badminton equipment purchase --/
theorem badminton_cost_comparison 
  (x : ℝ) 
  (h_x : x ≥ 3) 
  (yA yB : ℝ) 
  (h_yA : yA = 32 * x + 320) 
  (h_yB : yB = 40 * x + 280) : 
  (yA = yB ↔ x = 5) ∧ 
  (3 ≤ x ∧ x < 5 → yB < yA) ∧ 
  (x > 5 → yA < yB) := by
sorry

end badminton_cost_comparison_l2076_207629


namespace unique_p_q_for_inequality_l2076_207610

theorem unique_p_q_for_inequality :
  ∀ (p q : ℝ),
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |Real.sqrt (1 - x^2) - p*x - q| ≤ (Real.sqrt 2 - 1) / 2) →
    p = -1 ∧ q = (1 + Real.sqrt 2) / 2 := by
  sorry

end unique_p_q_for_inequality_l2076_207610


namespace line_slope_l2076_207605

/-- The slope of the line defined by the equation x/4 + y/3 = 1 is -3/4 -/
theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) :=
by sorry

end line_slope_l2076_207605


namespace sum_ratio_l2076_207641

def sean_sum : ℕ → ℕ
| 0 => 0
| (n + 1) => sean_sum n + if (n + 1) * 3 ≤ 600 then (n + 1) * 3 else 0

def julie_sum : ℕ → ℕ
| 0 => 0
| (n + 1) => julie_sum n + if n + 1 ≤ 300 then n + 1 else 0

theorem sum_ratio :
  (sean_sum 200 : ℚ) / (julie_sum 300 : ℚ) = 4 / 3 :=
by sorry

end sum_ratio_l2076_207641


namespace calculate_expression_l2076_207602

theorem calculate_expression : (3.242 * 12) / 100 = 0.38904 := by
  sorry

end calculate_expression_l2076_207602


namespace equation_solution_l2076_207687

theorem equation_solution (y : ℝ) : 
  (y^2 - 11*y + 24)/(y - 1) + (4*y^2 + 20*y - 25)/(4*y - 5) = 5 → y = 3 ∨ y = 4 := by
sorry

end equation_solution_l2076_207687


namespace colony_leadership_arrangements_l2076_207645

def colony_size : ℕ := 12
def num_deputies : ℕ := 2
def subordinates_per_deputy : ℕ := 3

def leadership_arrangements : ℕ :=
  colony_size *
  (colony_size - 1) *
  (colony_size - 2) *
  (Nat.choose (colony_size - num_deputies - 1) subordinates_per_deputy) *
  (Nat.choose (colony_size - num_deputies - 1 - subordinates_per_deputy) subordinates_per_deputy)

theorem colony_leadership_arrangements :
  leadership_arrangements = 2209600 :=
by sorry

end colony_leadership_arrangements_l2076_207645


namespace correct_number_of_plants_l2076_207693

/-- The number of large salads Anna needs -/
def salads_needed : ℕ := 12

/-- The fraction of lettuce that will survive (not lost to insects and rabbits) -/
def survival_rate : ℚ := 1/2

/-- The number of large salads each lettuce plant provides -/
def salads_per_plant : ℕ := 3

/-- The number of lettuce plants Anna should grow -/
def plants_to_grow : ℕ := 8

/-- Theorem stating that the number of plants Anna should grow is correct -/
theorem correct_number_of_plants : 
  plants_to_grow * salads_per_plant * survival_rate ≥ salads_needed := by
  sorry

end correct_number_of_plants_l2076_207693


namespace smallest_dual_palindrome_l2076_207650

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 8 → isPalindromeInBase n 2 → isPalindromeInBase n 8 → n ≥ 63 :=
by sorry

end smallest_dual_palindrome_l2076_207650


namespace circle_radius_proof_l2076_207608

/-- Given a circle with the following properties:
  - A chord of length 18
  - The chord is intersected by a diameter at a point
  - The intersection point is 7 units from the center
  - The intersection point divides the chord in the ratio 2:1
  Prove that the radius of the circle is 11 -/
theorem circle_radius_proof (chord_length : ℝ) (intersection_distance : ℝ) 
  (h1 : chord_length = 18)
  (h2 : intersection_distance = 7)
  (h3 : ∃ (a b : ℝ), a + b = chord_length ∧ a = 2 * b) :
  ∃ (radius : ℝ), radius = 11 ∧ radius^2 = intersection_distance^2 + (chord_length^2 / 4) :=
by sorry

end circle_radius_proof_l2076_207608


namespace circus_ticket_ratio_l2076_207603

theorem circus_ticket_ratio : 
  ∀ (num_kids num_adults : ℕ) 
    (total_cost kid_ticket_cost : ℚ),
  num_kids = 6 →
  num_adults = 2 →
  total_cost = 50 →
  kid_ticket_cost = 5 →
  (kid_ticket_cost / ((total_cost - num_kids * kid_ticket_cost) / num_adults)) = 1/2 := by
sorry

end circus_ticket_ratio_l2076_207603


namespace magic_box_theorem_l2076_207607

theorem magic_box_theorem (m : ℝ) : m^2 - 2*m - 1 = 2 → m = 3 ∨ m = -1 := by
  sorry

end magic_box_theorem_l2076_207607


namespace marias_score_is_correct_score_difference_average_score_correct_l2076_207634

/-- Maria's score in a game, given that it was 50 points more than John's and their average was 112 -/
def marias_score : ℕ := 137

/-- John's score in the game -/
def johns_score : ℕ := marias_score - 50

/-- The average score of Maria and John -/
def average_score : ℕ := 112

theorem marias_score_is_correct : marias_score = 137 := by
  sorry

theorem score_difference : marias_score = johns_score + 50 := by
  sorry

theorem average_score_correct : (marias_score + johns_score) / 2 = average_score := by
  sorry

end marias_score_is_correct_score_difference_average_score_correct_l2076_207634


namespace masters_percentage_is_76_l2076_207614

/-- Represents a sports team with juniors and masters -/
structure Team where
  juniors : ℕ
  masters : ℕ

/-- Calculates the percentage of masters in a team -/
def percentageMasters (team : Team) : ℚ :=
  (team.masters : ℚ) / ((team.juniors + team.masters) : ℚ) * 100

/-- Theorem stating that under the given conditions, the percentage of masters is 76% -/
theorem masters_percentage_is_76 (team : Team) 
  (h1 : 22 * team.juniors + 47 * team.masters = 41 * (team.juniors + team.masters)) :
  percentageMasters team = 76 := by
  sorry

#eval (76 : ℚ)

end masters_percentage_is_76_l2076_207614


namespace remaining_files_indeterminate_l2076_207624

/-- Represents the state of Dave's phone -/
structure PhoneState where
  apps : ℕ
  files : ℕ

/-- Represents the change in Dave's phone state -/
structure PhoneStateChange where
  initialState : PhoneState
  finalState : PhoneState
  appsDeleted : ℕ

/-- Predicate to check if a PhoneStateChange is valid according to the problem conditions -/
def isValidPhoneStateChange (change : PhoneStateChange) : Prop :=
  change.initialState.apps = 16 ∧
  change.initialState.files = 77 ∧
  change.finalState.apps = 5 ∧
  change.appsDeleted = 11 ∧
  change.initialState.apps - change.appsDeleted = change.finalState.apps ∧
  change.finalState.files ≤ change.initialState.files

/-- Theorem stating that the number of remaining files cannot be uniquely determined -/
theorem remaining_files_indeterminate (change : PhoneStateChange) 
  (h : isValidPhoneStateChange change) :
  ∃ (x y : ℕ), x ≠ y ∧ 
    isValidPhoneStateChange { change with finalState := { change.finalState with files := x } } ∧
    isValidPhoneStateChange { change with finalState := { change.finalState with files := y } } :=
  sorry

end remaining_files_indeterminate_l2076_207624


namespace square_in_S_l2076_207657

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

def S : Set ℕ :=
  {n | is_sum_of_two_squares (n - 1) ∧ 
       is_sum_of_two_squares n ∧ 
       is_sum_of_two_squares (n + 1)}

theorem square_in_S (n : ℕ) (hn : n ∈ S) : n^2 ∈ S := by
  sorry

end square_in_S_l2076_207657


namespace rotated_square_top_vertex_distance_l2076_207630

/-- The distance of the top vertex of a rotated square from the base line -/
theorem rotated_square_top_vertex_distance 
  (square_side : ℝ) 
  (rotation_angle : ℝ) :
  square_side = 2 →
  rotation_angle = π/4 →
  let diagonal := square_side * Real.sqrt 2
  let center_height := square_side / 2
  let vertical_shift := Real.sqrt 2 / 2 * diagonal
  center_height + vertical_shift = 1 + Real.sqrt 2 := by
  sorry

end rotated_square_top_vertex_distance_l2076_207630


namespace square_sum_equals_25_l2076_207683

theorem square_sum_equals_25 (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) :
  x^2 + y^2 = 25 := by
sorry

end square_sum_equals_25_l2076_207683
