import Mathlib

namespace problem_1_l1416_141671

theorem problem_1 : (1) - 8 + 12 - 16 - 23 = -35 := by
  sorry

end problem_1_l1416_141671


namespace sandhill_football_club_members_l1416_141639

/-- Represents the Sandhill Football Club problem --/
theorem sandhill_football_club_members :
  let sock_cost : ℕ := 5
  let tshirt_cost : ℕ := sock_cost + 6
  let home_game_socks : ℕ := 1
  let home_game_tshirts : ℕ := 1
  let away_game_socks : ℕ := 2
  let away_game_tshirts : ℕ := 1
  let total_expenditure : ℕ := 4150
  let member_cost : ℕ := 
    (home_game_socks + away_game_socks) * sock_cost + 
    (home_game_tshirts + away_game_tshirts) * tshirt_cost
  let number_of_members : ℕ := total_expenditure / member_cost
  number_of_members = 112 :=
by
  sorry


end sandhill_football_club_members_l1416_141639


namespace stock_order_l1416_141695

def initial_investment : ℝ := 100

def apple_year1 : ℝ := 1.50
def apple_year2 : ℝ := 0.75
def banana_year1 : ℝ := 0.50
def banana_year2 : ℝ := 2.00
def cherry_year1 : ℝ := 1.30
def cherry_year2 : ℝ := 1.10
def date_year1 : ℝ := 1.00
def date_year2 : ℝ := 0.80

def final_value (year1 : ℝ) (year2 : ℝ) : ℝ :=
  initial_investment * year1 * year2

theorem stock_order :
  let A := final_value apple_year1 apple_year2
  let B := final_value banana_year1 banana_year2
  let C := final_value cherry_year1 cherry_year2
  let D := final_value date_year1 date_year2
  D < B ∧ B < A ∧ A < C := by
  sorry

end stock_order_l1416_141695


namespace amanda_marbles_l1416_141655

theorem amanda_marbles (katrina_marbles : ℕ) (amanda_marbles : ℕ) (mabel_marbles : ℕ) : 
  mabel_marbles = 5 * katrina_marbles →
  mabel_marbles = 85 →
  mabel_marbles = amanda_marbles + 63 →
  2 * katrina_marbles - amanda_marbles = 12 :=
by
  sorry

end amanda_marbles_l1416_141655


namespace simplify_expression_l1416_141675

theorem simplify_expression (w : ℝ) : 3*w + 4 - 2*w - 5 + 6*w + 7 - 3*w - 9 = 4*w - 3 := by
  sorry

end simplify_expression_l1416_141675


namespace range_of_k_l1416_141640

theorem range_of_k (n : ℕ+) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
   |x₁ - 2*n| = k ∧ |x₂ - 2*n| = k) →
  0 < k ∧ k ≤ 1 :=
by sorry

end range_of_k_l1416_141640


namespace max_value_of_f_min_value_of_expression_equality_condition_l1416_141682

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x|

-- Theorem for the maximum value of f
theorem max_value_of_f : ∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ ∃ (y : ℝ), f y = m :=
sorry

-- Theorem for the minimum value of the expression
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 / (b + 1)) + (b^2 / (a + 1)) ≥ 1/3 :=
sorry

-- Theorem for the equality condition
theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 / (b + 1)) + (b^2 / (a + 1)) = 1/3 ↔ a = 1/2 ∧ b = 1/2 :=
sorry

end max_value_of_f_min_value_of_expression_equality_condition_l1416_141682


namespace power_sum_negative_two_l1416_141606

theorem power_sum_negative_two : (-2)^2002 + (-2)^2003 = -2^2002 := by
  sorry

end power_sum_negative_two_l1416_141606


namespace sum_of_odd_and_five_times_odd_is_even_l1416_141650

theorem sum_of_odd_and_five_times_odd_is_even (m n : ℕ) 
  (hm : m % 2 = 1) (hn : n % 2 = 1) (hm_pos : 0 < m) (hn_pos : 0 < n) : 
  ∃ k : ℕ, m + 5 * n = 2 * k := by
sorry

end sum_of_odd_and_five_times_odd_is_even_l1416_141650


namespace zoe_correct_percentage_l1416_141616

theorem zoe_correct_percentage
  (total : ℝ)
  (chloe_alone : ℝ)
  (zoe_alone : ℝ)
  (amy_alone : ℝ)
  (together : ℝ)
  (chloe_correct_alone : ℝ)
  (chloe_correct_overall : ℝ)
  (zoe_correct_alone : ℝ)
  (trio_correct_together : ℝ)
  (h1 : chloe_alone = 0.4 * total)
  (h2 : zoe_alone = 0.3 * total)
  (h3 : amy_alone = 0.3 * total)
  (h4 : together = total - (chloe_alone + zoe_alone + amy_alone))
  (h5 : chloe_correct_alone = 0.8 * chloe_alone)
  (h6 : chloe_correct_overall = 0.88 * (chloe_alone + together))
  (h7 : zoe_correct_alone = 0.75 * zoe_alone)
  (h8 : trio_correct_together = 0.85 * together)
  : (zoe_correct_alone + trio_correct_together) / (zoe_alone + together) = 0.85 := by
  sorry

end zoe_correct_percentage_l1416_141616


namespace rectangular_hall_dimensions_l1416_141685

theorem rectangular_hall_dimensions (length width : ℝ) : 
  width = length / 2 → 
  length * width = 450 → 
  length - width = 15 :=
by sorry

end rectangular_hall_dimensions_l1416_141685


namespace original_ratio_l1416_141621

theorem original_ratio (x y : ℕ) (h1 : x = y + 5) (h2 : (x - 5) / (y - 5) = 5 / 4) : x / y = 6 / 5 := by
  sorry

end original_ratio_l1416_141621


namespace perfect_square_condition_l1416_141635

theorem perfect_square_condition (m : ℤ) : 
  (∀ x : ℤ, ∃ y : ℤ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = y^2) → 
  m = 196 := by
  sorry

end perfect_square_condition_l1416_141635


namespace quadratic_inequality_range_l1416_141687

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end quadratic_inequality_range_l1416_141687


namespace line_angle_inclination_l1416_141694

/-- The angle of inclination of a line given its equation and a point it passes through -/
def angleOfInclination (a m : ℝ) (h1 : m ≠ 0) (h2 : a + m - 2*a = 0) : ℝ :=
  135

/-- Theorem: The angle of inclination of the line ax + my - 2a = 0 (m ≠ 0) passing through (1, 1) is 135° -/
theorem line_angle_inclination (a m : ℝ) (h1 : m ≠ 0) (h2 : a + m - 2*a = 0) :
  angleOfInclination a m h1 h2 = 135 := by
  sorry

end line_angle_inclination_l1416_141694


namespace f_inequality_iff_a_condition_l1416_141672

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - (a + 1) * Real.log x

-- State the theorem
theorem f_inequality_iff_a_condition (a : ℝ) :
  (∀ x > 0, f a x ≤ x) ↔ a ≥ -1 / (Real.exp 1 - 1) := by sorry

end f_inequality_iff_a_condition_l1416_141672


namespace isosceles_triangle_exists_l1416_141601

/-- A regular polygon with 101 vertices -/
structure RegularPolygon101 where
  vertices : Fin 101 → ℝ × ℝ

/-- A selection of 51 vertices from a 101-regular polygon -/
structure Selection51 (polygon : RegularPolygon101) where
  selected : Fin 51 → Fin 101
  distinct : ∀ i j, i ≠ j → selected i ≠ selected j

/-- Three points form an isosceles triangle -/
def IsIsoscelesTriangle (a b c : ℝ × ℝ) : Prop :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = (a.1 - c.1)^2 + (a.2 - c.2)^2 ∨
  (b.1 - a.1)^2 + (b.2 - a.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∨
  (c.1 - a.1)^2 + (c.2 - a.2)^2 = (c.1 - b.1)^2 + (c.2 - b.2)^2

/-- Main theorem: Among any 51 vertices of the 101-regular polygon, 
    there are three that form an isosceles triangle -/
theorem isosceles_triangle_exists (polygon : RegularPolygon101) 
  (selection : Selection51 polygon) : 
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    IsIsoscelesTriangle 
      (polygon.vertices (selection.selected i))
      (polygon.vertices (selection.selected j))
      (polygon.vertices (selection.selected k)) := by
  sorry

end isosceles_triangle_exists_l1416_141601


namespace chi_square_independence_hypothesis_l1416_141603

/-- Represents a χ² test of independence -/
structure ChiSquareTest where
  /-- The statistical hypothesis of the test -/
  hypothesis : Prop

/-- Represents events in a statistical context -/
structure Event

/-- Defines mutual independence for a list of events -/
def mutually_independent (events : List Event) : Prop :=
  sorry -- Definition of mutual independence

/-- The χ² test of independence assumes mutual independence of events -/
theorem chi_square_independence_hypothesis :
  ∀ (test : ChiSquareTest) (events : List Event),
    test.hypothesis ↔ mutually_independent events := by
  sorry

end chi_square_independence_hypothesis_l1416_141603


namespace john_notebooks_correct_l1416_141649

/-- The number of notebooks John bought -/
def notebooks : ℕ := 5

/-- The number of pages in each notebook -/
def pages_per_notebook : ℕ := 40

/-- The number of pages John uses per day -/
def pages_per_day : ℕ := 4

/-- The number of days the notebooks last -/
def days : ℕ := 50

/-- Theorem stating that the number of notebooks John bought is correct -/
theorem john_notebooks_correct : 
  notebooks * pages_per_notebook = pages_per_day * days := by
  sorry


end john_notebooks_correct_l1416_141649


namespace largest_sulfuric_acid_percentage_l1416_141651

/-- Represents the largest integer percentage of sulfuric acid solution that can be achieved in the first vessel after transfer -/
def largest_integer_percentage : ℕ := 76

/-- Represents the initial volume of solution in the first vessel -/
def initial_volume_1 : ℚ := 4

/-- Represents the initial volume of solution in the second vessel -/
def initial_volume_2 : ℚ := 3

/-- Represents the initial concentration of sulfuric acid in the first vessel -/
def initial_concentration_1 : ℚ := 70 / 100

/-- Represents the initial concentration of sulfuric acid in the second vessel -/
def initial_concentration_2 : ℚ := 90 / 100

/-- Represents the capacity of each vessel -/
def vessel_capacity : ℚ := 6

theorem largest_sulfuric_acid_percentage :
  ∀ x : ℚ,
  0 ≤ x ∧ x ≤ initial_volume_2 →
  (initial_volume_1 * initial_concentration_1 + x * initial_concentration_2) / (initial_volume_1 + x) ≤ largest_integer_percentage / 100 ∧
  ∃ y : ℚ, 0 < y ∧ y ≤ initial_volume_2 ∧
  (initial_volume_1 * initial_concentration_1 + y * initial_concentration_2) / (initial_volume_1 + y) > (largest_integer_percentage - 1) / 100 ∧
  initial_volume_1 + y ≤ vessel_capacity :=
by sorry

#check largest_sulfuric_acid_percentage

end largest_sulfuric_acid_percentage_l1416_141651


namespace mirror_area_l1416_141660

/-- Calculates the area of a rectangular mirror inside a frame -/
theorem mirror_area (frame_width frame_height frame_thickness : ℕ) : 
  frame_width = 65 ∧ frame_height = 85 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 1925 := by
  sorry

end mirror_area_l1416_141660


namespace min_value_of_roots_squared_difference_l1416_141612

theorem min_value_of_roots_squared_difference (a : ℝ) (m n : ℝ) 
  (h1 : a ≥ 1)
  (h2 : m^2 - 2*a*m + 1 = 0)
  (h3 : n^2 - 2*a*n + 1 = 0) :
  ∃ (k : ℝ), k = (m - 1)^2 + (n - 1)^2 ∧ k ≥ 0 ∧ ∀ (x : ℝ), x = (m - 1)^2 + (n - 1)^2 → x ≥ k :=
by sorry

end min_value_of_roots_squared_difference_l1416_141612


namespace rectangle_forms_same_solid_l1416_141661

-- Define the shapes
inductive Shape
  | RightTriangle
  | Rectangle
  | RightTrapezoid
  | IsoscelesRightTriangle

-- Define a function that determines if a shape forms the same solid when rotated around any edge
def forms_same_solid (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | _ => false

-- Theorem statement
theorem rectangle_forms_same_solid :
  ∀ s : Shape, forms_same_solid s ↔ s = Shape.Rectangle :=
by sorry

end rectangle_forms_same_solid_l1416_141661


namespace smaller_cube_edge_length_l1416_141647

/-- Given a cube with edge length 7 cm that is cut into smaller cubes, 
    if the total surface area increases by 600%, 
    then the edge length of the smaller cubes is 1 cm. -/
theorem smaller_cube_edge_length 
  (original_edge : ℝ) 
  (surface_area_increase : ℝ) 
  (smaller_edge : ℝ) : 
  original_edge = 7 →
  surface_area_increase = 6 →
  (6 * smaller_edge^2) * ((original_edge^3) / smaller_edge^3) = 
    (1 + surface_area_increase) * (6 * original_edge^2) →
  smaller_edge = 1 := by
sorry

end smaller_cube_edge_length_l1416_141647


namespace dan_catches_cate_l1416_141626

/-- The time it takes for Dan to catch Cate given their initial distance and speeds -/
theorem dan_catches_cate (initial_distance : ℝ) (dan_speed : ℝ) (cate_speed : ℝ)
  (h1 : initial_distance = 50)
  (h2 : dan_speed = 8)
  (h3 : cate_speed = 6)
  (h4 : dan_speed > cate_speed) :
  (initial_distance / (dan_speed - cate_speed)) = 25 :=
by sorry

end dan_catches_cate_l1416_141626


namespace point_movement_theorem_l1416_141691

/-- The initial position of a point on a number line that ends at the origin after moving right 7 units and then left 4 units -/
def initial_position : ℤ := -3

/-- A point's movement on a number line -/
def point_movement (start : ℤ) : ℤ := start + 7 - 4

theorem point_movement_theorem :
  point_movement initial_position = 0 :=
by sorry

end point_movement_theorem_l1416_141691


namespace quadratic_circle_theorem_l1416_141673

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + b

-- Define the condition that f intersects the axes at three points
def intersects_axes_at_three_points (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f b x₁ = 0 ∧ f b x₂ = 0 ∧ f b 0 ≠ 0

-- Define the equation of the circle
def circle_equation (b : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + 2*x - (b + 1)*y + b

-- Main theorem
theorem quadratic_circle_theorem (b : ℝ) 
  (h : intersects_axes_at_three_points b) :
  (b < 1 ∧ b ≠ 0) ∧
  (∀ x y : ℝ, circle_equation b x y = 0 ↔ 
    (x = 0 ∧ f b y = 0) ∨ (y = 0 ∧ f b x = 0) ∨ (x = 0 ∧ y = b)) ∧
  (circle_equation b 0 1 = 0 ∧ circle_equation b (-2) 1 = 0) :=
sorry

end quadratic_circle_theorem_l1416_141673


namespace convex_quadrilaterals_from_circle_points_l1416_141653

theorem convex_quadrilaterals_from_circle_points (n : ℕ) (h : n = 20) :
  Nat.choose n 4 = 4845 :=
sorry

end convex_quadrilaterals_from_circle_points_l1416_141653


namespace movie_theater_seats_l1416_141625

theorem movie_theater_seats (total_seats : ℕ) (num_sections : ℕ) (seats_per_section : ℕ) :
  total_seats = 270 → num_sections = 9 → total_seats = num_sections * seats_per_section →
  seats_per_section = 30 := by
  sorry

end movie_theater_seats_l1416_141625


namespace complement_union_theorem_l1416_141662

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem complement_union_theorem : 
  (U \ M) ∪ (U \ N) = {2, 3, 4, 5} := by sorry

end complement_union_theorem_l1416_141662


namespace unique_solution_exists_l1416_141609

/-- Predicate to check if exactly one of three numbers is negative -/
def exactlyOneNegative (a b c : ℝ) : Prop :=
  (a < 0 ∧ b > 0 ∧ c > 0) ∨ (a > 0 ∧ b < 0 ∧ c > 0) ∨ (a > 0 ∧ b > 0 ∧ c < 0)

/-- The main theorem stating that there exists exactly one solution -/
theorem unique_solution_exists :
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a = Real.sqrt (b * c) ∧
    b = Real.sqrt (c * a) ∧
    c = Real.sqrt (a * b) ∧
    exactlyOneNegative a b c :=
by
  sorry

end unique_solution_exists_l1416_141609


namespace integer_equation_solution_l1416_141666

theorem integer_equation_solution (x y : ℤ) : 
  x^2 = y^2 + 2*y + 13 ↔ (x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = 1) ∨ (x = 4 ∧ y = -3) ∨ (x = -4 ∧ y = -3) := by
  sorry

end integer_equation_solution_l1416_141666


namespace zachary_pushups_l1416_141619

theorem zachary_pushups (zachary : ℕ) (david : ℕ) : 
  david = zachary + 58 → 
  zachary + david = 146 → 
  zachary = 44 := by
sorry

end zachary_pushups_l1416_141619


namespace book_pages_theorem_l1416_141620

/-- Represents a reading pattern of a book -/
structure ReadingPattern where
  first_day : ℕ
  daily_increase : ℕ
  pages_left : ℕ

/-- Calculates the total number of pages in a book based on two reading patterns -/
def calculate_total_pages (r1 r2 : ReadingPattern) : ℕ :=
  sorry

/-- The theorem stating the total number of pages in the book -/
theorem book_pages_theorem (r1 r2 : ReadingPattern) 
  (h1 : r1.first_day = 35 ∧ r1.daily_increase = 5 ∧ r1.pages_left = 35)
  (h2 : r2.first_day = 45 ∧ r2.daily_increase = 5 ∧ r2.pages_left = 40) :
  calculate_total_pages r1 r2 = 385 :=
sorry

end book_pages_theorem_l1416_141620


namespace parabola_focus_directrix_distance_l1416_141615

/-- A parabola is defined by its equation y² = 2x -/
def Parabola := {(x, y) : ℝ × ℝ | y^2 = 2*x}

/-- The distance from the focus to the directrix of a parabola y² = 2x is 1 -/
theorem parabola_focus_directrix_distance :
  ∃ (f d : ℝ × ℝ), f ∈ Parabola ∧ (∀ (p : ℝ × ℝ), p ∈ Parabola → dist p d = dist p f) ∧ dist f d = 1 :=
sorry

end parabola_focus_directrix_distance_l1416_141615


namespace total_pencil_length_l1416_141629

/-- The length of Isha's first pencil in cubes -/
def first_pencil_cubes : ℕ := 12

/-- The length of each cube in Isha's first pencil in centimeters -/
def first_pencil_cube_length : ℚ := 3/2

/-- The length of the second pencil in cubes -/
def second_pencil_cubes : ℕ := 13

/-- The length of each cube in the second pencil in centimeters -/
def second_pencil_cube_length : ℚ := 17/10

/-- The total length of both pencils in centimeters -/
def total_length : ℚ := first_pencil_cubes * first_pencil_cube_length + 
                        second_pencil_cubes * second_pencil_cube_length

theorem total_pencil_length : total_length = 401/10 := by
  sorry

end total_pencil_length_l1416_141629


namespace max_value_abc_max_value_abc_attained_l1416_141674

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  a^3 * b^2 * c^2 ≤ 432 / 7^7 := by
  sorry

theorem max_value_abc_attained (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 1 ∧ a₀^3 * b₀^2 * c₀^2 = 432 / 7^7 := by
  sorry

end max_value_abc_max_value_abc_attained_l1416_141674


namespace possible_values_of_a_l1416_141665

def A (a : ℝ) := {x : ℝ | 0 < x ∧ x < a}
def B := {x : ℝ | 1 < x ∧ x < 2}

theorem possible_values_of_a (a : ℝ) :
  (A a).Nonempty ∧ B ⊆ (Aᶜ a) ↔ 0 < a ∧ a ≤ 1 :=
sorry

end possible_values_of_a_l1416_141665


namespace polynomial_has_solution_mod_prime_l1416_141678

/-- The polynomial f(x) = x^6 - 11x^4 + 36x^2 - 36 -/
def f (x : ℤ) : ℤ := x^6 - 11*x^4 + 36*x^2 - 36

/-- For any prime p, there exists an x such that f(x) ≡ 0 (mod p) -/
theorem polynomial_has_solution_mod_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ x : ℤ, f x ≡ 0 [ZMOD p] := by sorry

end polynomial_has_solution_mod_prime_l1416_141678


namespace equation_pattern_l1416_141677

theorem equation_pattern (n : ℕ) : 2*n * (2*n + 2) + 1 = (2*n + 1)^2 := by
  sorry

end equation_pattern_l1416_141677


namespace grocery_cost_l1416_141618

/-- The cost of groceries problem -/
theorem grocery_cost (mango_price rice_price flour_price : ℝ) 
  (h1 : 10 * mango_price = 24 * rice_price)
  (h2 : flour_price = 2 * rice_price)
  (h3 : flour_price = 20.50) : 
  4 * mango_price + 3 * rice_price + 5 * flour_price = 231.65 := by
  sorry

end grocery_cost_l1416_141618


namespace sculpture_exposed_area_l1416_141644

/-- Represents a layer in the sculpture -/
structure Layer where
  cubes : ℕ
  exposed_top : ℕ
  exposed_side : ℕ

/-- The sculpture configuration -/
def sculpture : List Layer := [
  ⟨9, 9, 16⟩,
  ⟨6, 6, 10⟩,
  ⟨4, 4, 8⟩,
  ⟨1, 1, 4⟩
]

/-- The total number of cubes in the sculpture -/
def total_cubes : ℕ := (sculpture.map Layer.cubes).sum

/-- Calculates the exposed surface area of a layer -/
def exposed_area (layer : Layer) : ℕ := layer.exposed_top + layer.exposed_side

/-- Calculates the total exposed surface area of the sculpture -/
def total_exposed_area : ℕ := (sculpture.map exposed_area).sum

/-- Theorem: The total exposed surface area of the sculpture is 58 square meters -/
theorem sculpture_exposed_area :
  total_cubes = 20 ∧ total_exposed_area = 58 := by sorry

end sculpture_exposed_area_l1416_141644


namespace lowest_sale_price_percentage_l1416_141652

theorem lowest_sale_price_percentage (list_price : ℝ) (regular_discount_max : ℝ) (additional_discount : ℝ) : 
  list_price = 80 →
  regular_discount_max = 0.5 →
  additional_discount = 0.2 →
  (list_price * (1 - regular_discount_max) - list_price * additional_discount) / list_price = 0.3 := by
sorry

end lowest_sale_price_percentage_l1416_141652


namespace equal_real_imag_parts_l1416_141646

theorem equal_real_imag_parts (b : ℝ) : 
  let z : ℂ := (1 + I) / (1 - I) + (1 / 2 : ℂ) * b
  (z.re = z.im) ↔ b = 2 := by sorry

end equal_real_imag_parts_l1416_141646


namespace c_class_size_l1416_141681

/-- The number of students in each class -/
structure ClassSize where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : ClassSize) : Prop :=
  s.A = 44 ∧ s.B = s.A + 2 ∧ s.C = s.B - 1

/-- The theorem to prove -/
theorem c_class_size (s : ClassSize) (h : problem_conditions s) : s.C = 45 := by
  sorry


end c_class_size_l1416_141681


namespace trays_from_second_table_l1416_141611

theorem trays_from_second_table 
  (trays_per_trip : ℕ) 
  (total_trips : ℕ) 
  (trays_first_table : ℕ) 
  (h1 : trays_per_trip = 4) 
  (h2 : total_trips = 9) 
  (h3 : trays_first_table = 20) : 
  trays_per_trip * total_trips - trays_first_table = 16 := by
  sorry

end trays_from_second_table_l1416_141611


namespace train_speed_l1416_141692

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) :
  train_length = 300 →
  crossing_time = 17.998560115190784 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63.00468) < 0.00001 := by
  sorry


end train_speed_l1416_141692


namespace jeff_travel_distance_l1416_141632

/-- Calculates the total distance traveled given a list of speed-time pairs -/
def totalDistance (segments : List (Real × Real)) : Real :=
  segments.foldl (fun acc (speed, time) => acc + speed * time) 0

/-- Proves that Jeff's total travel distance is 820 miles -/
theorem jeff_travel_distance :
  let segments : List (Real × Real) := [
    (80, 3), (50, 2), (70, 1), (60, 2),
    (45, 3), (40, 2), (30, 2.5)
  ]
  totalDistance segments = 820 := by
  sorry

#eval totalDistance [(80, 3), (50, 2), (70, 1), (60, 2), (45, 3), (40, 2), (30, 2.5)]

end jeff_travel_distance_l1416_141632


namespace pascal_triangle_elements_l1416_141676

/-- The number of elements in a single row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def totalElementsPascal (n : ℕ) : ℕ := triangularNumber n

theorem pascal_triangle_elements :
  totalElementsPascal 30 = 465 := by
  sorry

end pascal_triangle_elements_l1416_141676


namespace polyhedron_special_value_l1416_141638

/-- Represents a convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  T : ℕ  -- Number of triangular faces meeting at each vertex
  P : ℕ  -- Number of pentagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 32
  vertex_face_relation : V * (P / 5 + T / 3 : ℚ) = 32

/-- Theorem stating the specific value of 100P + 10T + V for the given polyhedron -/
theorem polyhedron_special_value (poly : ConvexPolyhedron) : 
  100 * poly.P + 10 * poly.T + poly.V = 250 := by
  sorry


end polyhedron_special_value_l1416_141638


namespace cost_splitting_difference_l1416_141690

def bob_paid : ℚ := 130
def alice_paid : ℚ := 110
def jessica_paid : ℚ := 160

def total_paid : ℚ := bob_paid + alice_paid + jessica_paid
def share_per_person : ℚ := total_paid / 3

def bob_owes : ℚ := share_per_person - bob_paid
def alice_owes : ℚ := share_per_person - alice_paid
def jessica_receives : ℚ := jessica_paid - share_per_person

theorem cost_splitting_difference :
  bob_owes - alice_owes = -20 := by sorry

end cost_splitting_difference_l1416_141690


namespace sqrt_360000_equals_600_l1416_141641

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_equals_600_l1416_141641


namespace light_ray_distance_l1416_141602

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

-- Define the foci of the ellipse
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define the total distance traveled by the light ray
def total_distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem light_ray_distance :
  ∀ p q : ℝ × ℝ,
  ellipse p.1 p.2 →
  ellipse q.1 q.2 →
  total_distance left_focus p + total_distance p right_focus +
  total_distance right_focus q + total_distance q left_focus = 12 :=
sorry

end light_ray_distance_l1416_141602


namespace bulls_win_probability_l1416_141670

/-- The probability of the Knicks winning a single game -/
def p_knicks : ℚ := 3/5

/-- The probability of the Bulls winning a single game -/
def p_bulls : ℚ := 1 - p_knicks

/-- The number of ways to choose 3 games out of 6 -/
def ways_to_choose : ℕ := 20

/-- The probability of the Bulls winning the playoff series in exactly 7 games -/
def prob_bulls_win_in_seven : ℚ :=
  ways_to_choose * p_bulls^3 * p_knicks^3 * p_bulls

theorem bulls_win_probability :
  prob_bulls_win_in_seven = 864/15625 := by sorry

end bulls_win_probability_l1416_141670


namespace equation_solution_l1416_141630

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (6 * x)^5 = (18 * x)^4 → x = 27 / 2 := by
  sorry

end equation_solution_l1416_141630


namespace smallest_integer_with_given_remainders_l1416_141659

theorem smallest_integer_with_given_remainders : ∃ (b : ℕ), 
  b > 0 ∧ 
  b % 3 = 0 ∧ 
  b % 4 = 2 ∧ 
  b % 5 = 3 ∧ 
  ∀ (n : ℕ), n > 0 ∧ n % 3 = 0 ∧ n % 4 = 2 ∧ n % 5 = 3 → b ≤ n :=
by
  -- The proof goes here
  sorry

end smallest_integer_with_given_remainders_l1416_141659


namespace min_box_height_is_ten_l1416_141669

/-- Represents the side length of the square base of the box -/
def base_side : ℝ → ℝ := λ x => x

/-- Represents the height of the box -/
def box_height : ℝ → ℝ := λ x => x + 5

/-- Calculates the surface area of the box -/
def surface_area : ℝ → ℝ := λ x => 2 * x^2 + 4 * x * (x + 5)

/-- Theorem: The minimum height of the box satisfying the given conditions is 10 units -/
theorem min_box_height_is_ten :
  ∃ (x : ℝ), x > 0 ∧ 
             surface_area x ≥ 130 ∧ 
             box_height x = 10 ∧
             ∀ (y : ℝ), y > 0 ∧ surface_area y ≥ 130 → box_height y ≥ box_height x :=
by sorry


end min_box_height_is_ten_l1416_141669


namespace jacket_purchase_price_l1416_141683

/-- Proves that the purchase price of a jacket is $56 given the specified conditions --/
theorem jacket_purchase_price :
  ∀ (purchase_price selling_price sale_price : ℝ),
  selling_price = purchase_price + 0.3 * selling_price →
  sale_price = 0.8 * selling_price →
  sale_price - purchase_price = 8 →
  purchase_price = 56 := by
sorry

end jacket_purchase_price_l1416_141683


namespace total_points_scored_l1416_141668

theorem total_points_scored (team_a team_b team_c : ℕ) 
  (h1 : team_a = 2) 
  (h2 : team_b = 9) 
  (h3 : team_c = 4) : 
  team_a + team_b + team_c = 15 := by
  sorry

end total_points_scored_l1416_141668


namespace group_purchase_equation_l1416_141696

/-- Represents a group purchase scenario where:
    - x is the number of people
    - p is the price of the item in coins
    - If each person contributes 8 coins, there's an excess of 3 coins
    - If each person contributes 7 coins, there's a shortage of 4 coins -/
structure GroupPurchase where
  x : ℕ  -- number of people
  p : ℕ  -- price of the item in coins
  excess_condition : 8 * x = p + 3
  shortage_condition : 7 * x + 4 = p

/-- Theorem stating that in a valid GroupPurchase scenario, 
    the number of people satisfies the equation 8x - 3 = 7x + 4 -/
theorem group_purchase_equation (gp : GroupPurchase) : 8 * gp.x - 3 = 7 * gp.x + 4 := by
  sorry


end group_purchase_equation_l1416_141696


namespace new_observations_sum_l1416_141654

theorem new_observations_sum (initial_count : ℕ) (initial_avg : ℚ) (new_count : ℕ) (new_avg : ℚ) :
  initial_count = 9 →
  initial_avg = 15 →
  new_count = 3 →
  new_avg = 13 →
  (initial_count * initial_avg + new_count * (3 * new_avg - initial_count * initial_avg)) / new_count = 21 :=
by sorry

end new_observations_sum_l1416_141654


namespace class_composition_l1416_141643

theorem class_composition (n : ℕ) (m : ℕ) : 
  n > 0 ∧ m > 0 ∧ m ≤ n ∧ 
  (⌊(m : ℚ) / n * 100 + 0.5⌋ : ℚ) = 51 →
  Odd n ∧ n ≥ 35 :=
by sorry

end class_composition_l1416_141643


namespace circular_saw_blade_distance_l1416_141663

/-- Given a circle with center (2, 3) and radius 8, and points A, B, and C on the circle
    such that ∠ABC is a right angle, AB = 8, and BC = 3, 
    prove that the square of the distance from B to the center of the circle is 41. -/
theorem circular_saw_blade_distance (A B C : ℝ × ℝ) : 
  let O : ℝ × ℝ := (2, 3)
  let r : ℝ := 8
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 →  -- A is on the circle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 →  -- B is on the circle
  (C.1 - O.1)^2 + (C.2 - O.2)^2 = r^2 →  -- C is on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8^2 →  -- AB = 8
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 3^2 →  -- BC = 3
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →  -- ∠ABC is a right angle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 41 := by
sorry

end circular_saw_blade_distance_l1416_141663


namespace outfit_combinations_l1416_141698

/-- The number of shirts -/
def num_shirts : ℕ := 4

/-- The number of pants -/
def num_pants : ℕ := 5

/-- The number of items (shirts or pants) that have a unique color -/
def num_unique_colors : ℕ := num_shirts + num_pants - 1

/-- The number of different outfits that can be created -/
def num_outfits : ℕ := num_shirts * num_pants - 1

theorem outfit_combinations : num_outfits = 19 := by sorry

end outfit_combinations_l1416_141698


namespace triangle_side_length_l1416_141600

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  b = 7 →
  c = 6 →
  Real.cos (B - C) = 31/32 →
  a = (Real.sqrt 299) / 2 :=
by sorry

end triangle_side_length_l1416_141600


namespace quadratic_equation_general_form_l1416_141608

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (4 * x = x^2 - 8) ↔ (x^2 - 4*x - 8 = 0) := by
sorry

end quadratic_equation_general_form_l1416_141608


namespace minimum_orange_chips_l1416_141607

theorem minimum_orange_chips 
  (purple green orange : ℕ) 
  (h1 : green ≥ purple / 3)
  (h2 : green ≤ orange / 4)
  (h3 : purple + green ≥ 75) :
  orange ≥ 76 := by
  sorry

end minimum_orange_chips_l1416_141607


namespace man_speed_against_current_l1416_141631

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that for the given speeds, the man's speed against the current is 9.6 km/hr -/
theorem man_speed_against_current :
  speed_against_current 16 3.2 = 9.6 := by
  sorry

#eval speed_against_current 16 3.2

end man_speed_against_current_l1416_141631


namespace cubic_roots_problem_l1416_141622

theorem cubic_roots_problem (u v c d : ℝ) : 
  (∃ w, {u, v, w} = {x | x^3 + c*x + d = 0}) ∧
  (∃ w', {u+5, v-4, w'} = {x | x^3 + c*x + (d+300) = 0}) →
  d = -4 ∨ d = 6 := by
sorry

end cubic_roots_problem_l1416_141622


namespace bowl_weight_after_refill_l1416_141623

theorem bowl_weight_after_refill (empty_bowl_weight : ℕ) 
  (day1_food day2_food day3_food day4_food : ℕ) :
  let total_food := day1_food + day2_food + day3_food + day4_food
  empty_bowl_weight + total_food = 
    empty_bowl_weight + day1_food + day2_food + day3_food + day4_food :=
by sorry

end bowl_weight_after_refill_l1416_141623


namespace pencil_black_fraction_l1416_141693

theorem pencil_black_fraction :
  ∀ (total_length blue_length white_length black_length : ℝ),
    total_length = 8 →
    blue_length = 3.5 →
    white_length = (total_length - blue_length) / 2 →
    black_length = total_length - blue_length - white_length →
    black_length / total_length = 9 / 32 := by
  sorry

end pencil_black_fraction_l1416_141693


namespace paving_cost_proof_l1416_141699

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Proof that the cost of paving the given floor is Rs. 28,875 -/
theorem paving_cost_proof :
  paving_cost 5.5 3.75 1400 = 28875 := by
  sorry

end paving_cost_proof_l1416_141699


namespace cubic_root_sum_squares_l1416_141624

theorem cubic_root_sum_squares (p q r : ℝ) (x : ℝ → ℝ) 
  (hx : ∀ t, x t = t^3 - p*t^2 + q*t - r) : 
  ∃ (r s t : ℝ), (x r = 0 ∧ x s = 0 ∧ x t = 0) ∧ 
  (r^2 + s^2 + t^2 = p^2 - 2*q) := by
  sorry

end cubic_root_sum_squares_l1416_141624


namespace hemisphere_surface_area_l1416_141657

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) :
  π * r^2 = 256 * π → 2 * π * r^2 + π * r^2 = 768 * π := by
  sorry

end hemisphere_surface_area_l1416_141657


namespace quadratic_real_roots_l1416_141664

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x - 1 = 0) ↔ (m ≥ -3 ∧ m ≠ 1) :=
by sorry

end quadratic_real_roots_l1416_141664


namespace statement_correctness_l1416_141627

theorem statement_correctness : 
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) ∧
  ¬(∀ a b : ℝ, (a > b ∧ b > 0 ↔ 1/a < 1/b)) ∧
  ¬(∀ a b : ℝ, (a > b ∧ b > 0 ↔ a^3 > b^3)) :=
by sorry

end statement_correctness_l1416_141627


namespace complex_norm_squared_l1416_141679

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 3 - 5*I) : 
  Complex.abs z^2 = 17/3 := by
sorry

end complex_norm_squared_l1416_141679


namespace f_derivative_inequality_implies_a_range_existence_of_intersecting_tangents_l1416_141642

/-- The function f(x) = x³ - ax² + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem f_derivative_inequality_implies_a_range :
  (∀ x : ℝ, f' 0 x ≥ |x| - 3/4) → -1 ≤ 0 ∧ 0 ≤ 1 := by sorry

theorem existence_of_intersecting_tangents :
  ∃ x₁ x₂ t : ℝ, x₁ ≠ x₂ ∧ 
  (f 0 x₁ + f' 0 x₁ * (2 - x₁) = t) ∧ 
  (f 0 x₂ + f' 0 x₂ * (2 - x₂) = t) ∧
  t ≤ 10 ∧
  (∀ s : ℝ, (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ 
    (f 0 y₁ + f' 0 y₁ * (2 - y₁) = s) ∧ 
    (f 0 y₂ + f' 0 y₂ * (2 - y₂) = s)) → 
  s ≤ 10) := by sorry

end f_derivative_inequality_implies_a_range_existence_of_intersecting_tangents_l1416_141642


namespace probability_of_blue_ball_l1416_141604

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of blue balls
def blue_balls : ℕ := 6

-- Define the probability of drawing a blue ball
def prob_blue_ball : ℚ := blue_balls / total_balls

-- Theorem statement
theorem probability_of_blue_ball : prob_blue_ball = 3/5 := by
  sorry

end probability_of_blue_ball_l1416_141604


namespace job_completion_time_l1416_141686

/-- The time taken for three workers to complete a job together, given their individual work rates -/
theorem job_completion_time 
  (rate_a rate_b rate_c : ℚ) 
  (h_a : rate_a = 1 / 8) 
  (h_b : rate_b = 1 / 16) 
  (h_c : rate_c = 1 / 16) : 
  1 / (rate_a + rate_b + rate_c) = 4 := by
  sorry

end job_completion_time_l1416_141686


namespace recyclable_containers_l1416_141645

theorem recyclable_containers (total_guests : ℕ) (soda_cans : ℕ) (water_bottles : ℕ) (juice_bottles : ℕ)
  (h_guests : total_guests = 90)
  (h_soda : soda_cans = 50)
  (h_water : water_bottles = 50)
  (h_juice : juice_bottles = 50)
  (h_soda_drinkers : total_guests / 2 = 45)
  (h_water_drinkers : total_guests / 3 = 30)
  (h_juice_consumed : juice_bottles * 4 / 5 = 40) :
  45 + 30 + 40 = 115 := by
  sorry

#check recyclable_containers

end recyclable_containers_l1416_141645


namespace cone_surface_area_l1416_141614

theorem cone_surface_area (r : ℝ) (h : r = 6) : 
  let sector_radius : ℝ := r
  let base_radius : ℝ := r / 2
  let slant_height : ℝ := sector_radius
  let base_area : ℝ := π * base_radius ^ 2
  let lateral_area : ℝ := π * base_radius * slant_height
  base_area + lateral_area = 27 * π := by sorry

end cone_surface_area_l1416_141614


namespace soccer_league_games_l1416_141637

theorem soccer_league_games (n : ℕ) (h : n = 14) : (n * (n - 1)) / 2 = 91 := by
  sorry

end soccer_league_games_l1416_141637


namespace logarithm_sum_equals_three_logarithm_base_25_144_l1416_141689

-- Part 1
theorem logarithm_sum_equals_three :
  (Real.log 2) ^ 2 + (Real.log 20 + 2) * Real.log 5 + Real.log 4 = 3 := by sorry

-- Part 2
theorem logarithm_base_25_144 (a b : ℝ) (h1 : Real.log 3 / Real.log 5 = a) (h2 : Real.log 4 / Real.log 5 = b) :
  Real.log 144 / Real.log 25 = a + b := by sorry

end logarithm_sum_equals_three_logarithm_base_25_144_l1416_141689


namespace solution_sum_l1416_141684

/-- Given a system of equations, prove that the sum of its solutions is 2020 -/
theorem solution_sum (x₀ y₀ : ℝ) 
  (eq1 : x₀^3 - 2023*x₀ = 2023*y₀ - y₀^3 - 2020)
  (eq2 : x₀^2 - x₀*y₀ + y₀^2 = 2022) :
  x₀ + y₀ = 2020 := by
sorry

end solution_sum_l1416_141684


namespace shopping_lottery_largest_number_l1416_141633

/-- Represents the largest number in a systematic sample -/
def largest_sample_number (total : ℕ) (start : ℕ) (interval : ℕ) : ℕ :=
  start + interval * ((total - start) / interval)

/-- The problem statement as a theorem -/
theorem shopping_lottery_largest_number :
  let total := 160
  let start := 7
  let second := 23
  let interval := second - start
  largest_sample_number total start interval = 151 := by
  sorry

#eval largest_sample_number 160 7 16

end shopping_lottery_largest_number_l1416_141633


namespace solution_set_when_m_2_solution_set_condition_l1416_141658

-- Define the function f
def f (x m : ℝ) : ℝ := |2*x - m| + 4*x

-- Part I
theorem solution_set_when_m_2 :
  {x : ℝ | f x 2 ≤ 1} = {x : ℝ | x ≤ -1/2} := by sorry

-- Part II
theorem solution_set_condition (m : ℝ) :
  {x : ℝ | f x m ≤ 2} = {x : ℝ | x ≤ -2} ↔ m = 6 ∨ m = -14 := by sorry

end solution_set_when_m_2_solution_set_condition_l1416_141658


namespace abs_comparison_negative_numbers_l1416_141688

theorem abs_comparison_negative_numbers (x y : ℝ) 
  (hx_neg : x < 0) (hy_neg : y < 0) (hxy : x < y) : 
  |x| > |y| := by
  sorry

end abs_comparison_negative_numbers_l1416_141688


namespace polynomial_division_theorem_l1416_141697

theorem polynomial_division_theorem (x : ℝ) :
  (x - 3) * (x^5 - 2*x^4 + 18*x^3 + 42*x^2 + 135*x + 387) + 1221 =
  x^6 - 5*x^5 + 24*x^4 - 12*x^3 + 9*x^2 - 18*x + 15 := by
  sorry

end polynomial_division_theorem_l1416_141697


namespace phone_number_probability_l1416_141680

theorem phone_number_probability (n : ℕ) (h : n = 10) :
  let p : ℚ := 1 / n
  1 - (1 - p) * (1 - p) = 1 / 5 :=
by sorry

end phone_number_probability_l1416_141680


namespace calculator_to_protractor_equivalence_l1416_141605

/-- Exchange rates at a math conference -/
structure ExchangeRates where
  calculator_to_ruler : ℚ
  ruler_to_compass : ℚ
  compass_to_protractor : ℚ

/-- The exchange rates given in the problem -/
def conference_rates : ExchangeRates where
  calculator_to_ruler := 100
  ruler_to_compass := 3/1
  compass_to_protractor := 2/1

/-- Theorem stating the equivalence between calculators and protractors -/
theorem calculator_to_protractor_equivalence (rates : ExchangeRates) :
  rates.calculator_to_ruler * rates.ruler_to_compass * rates.compass_to_protractor = 600 → 
  rates = conference_rates :=
sorry

#check calculator_to_protractor_equivalence

end calculator_to_protractor_equivalence_l1416_141605


namespace parking_lot_theorem_l1416_141617

/-- Represents a parking lot configuration --/
structure ParkingLot where
  grid : Fin 7 → Fin 7 → Bool
  gate : Fin 7 × Fin 7

/-- Checks if a car can exit from its position --/
def canExit (lot : ParkingLot) (pos : Fin 7 × Fin 7) : Bool :=
  sorry

/-- Counts the number of cars in the parking lot --/
def carCount (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if all cars in the lot can exit --/
def allCarsCanExit (lot : ParkingLot) : Bool :=
  sorry

/-- The maximum number of cars that can be parked --/
def maxCars : Nat := 28

theorem parking_lot_theorem (lot : ParkingLot) :
  (allCarsCanExit lot) → (carCount lot ≤ maxCars) :=
  sorry

end parking_lot_theorem_l1416_141617


namespace dwarf_attire_comparison_l1416_141613

/-- Represents the problem of comparing dwarf groups based on their attire. -/
theorem dwarf_attire_comparison :
  let total_dwarves : ℕ := 25
  let dwarves_without_hats : ℕ := 12
  let barefoot_dwarves : ℕ := 5
  let dwarves_with_hats := total_dwarves - dwarves_without_hats
  let dwarves_with_shoes := total_dwarves - barefoot_dwarves
  let dwarves_with_shoes_no_hat := dwarves_with_shoes - dwarves_with_hats
  dwarves_with_hats = dwarves_with_shoes_no_hat + 6 :=
by
  sorry


end dwarf_attire_comparison_l1416_141613


namespace smallest_factorization_coefficient_l1416_141667

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ r s : ℤ, ∀ x : ℤ, x^2 + b*x + 1800 = (x + r) * (x + s)) →
  b ≥ 85 :=
by sorry

end smallest_factorization_coefficient_l1416_141667


namespace neighborhood_total_l1416_141648

/-- Represents the number of households in different categories -/
structure Neighborhood where
  neither : ℕ
  both : ℕ
  with_car : ℕ
  bike_only : ℕ

/-- Calculates the total number of households in the neighborhood -/
def total_households (n : Neighborhood) : ℕ :=
  n.neither + (n.with_car - n.both) + n.bike_only + n.both

/-- Theorem stating that the total number of households is 90 -/
theorem neighborhood_total (n : Neighborhood) 
  (h1 : n.neither = 11)
  (h2 : n.both = 14)
  (h3 : n.with_car = 44)
  (h4 : n.bike_only = 35) : 
  total_households n = 90 := by
  sorry

#eval total_households { neither := 11, both := 14, with_car := 44, bike_only := 35 }

end neighborhood_total_l1416_141648


namespace angle_bisector_slope_l1416_141656

theorem angle_bisector_slope :
  let line1 : ℝ → ℝ := λ x => 2 * x
  let line2 : ℝ → ℝ := λ x => -2 * x
  let slope1 : ℝ := 2
  let slope2 : ℝ := -2
  let angle_bisector_slope : ℝ := (slope1 + slope2 + Real.sqrt (1 + slope1^2 + slope2^2)) / (1 - slope1 * slope2)
  angle_bisector_slope = 3/5 := by
  sorry

end angle_bisector_slope_l1416_141656


namespace seating_theorem_l1416_141610

/-- The number of ways to seat 9 children (5 sons and 4 daughters) in a row
    such that at least two girls are next to each other. -/
def seating_arrangements (total : ℕ) (sons : ℕ) (daughters : ℕ) : ℕ :=
  Nat.factorial total - (Nat.factorial sons * Nat.factorial daughters)

theorem seating_theorem :
  seating_arrangements 9 5 4 = 359400 := by
  sorry

end seating_theorem_l1416_141610


namespace students_in_both_competitions_l1416_141628

/-- The number of students who participated in both competitions A and B -/
def students_in_both (total students_A students_B : ℕ) : ℕ :=
  students_A + students_B - total

theorem students_in_both_competitions 
  (total : ℕ) (students_A : ℕ) (students_B : ℕ)
  (h_total : total = 55)
  (h_A : students_A = 38)
  (h_B : students_B = 42)
  (h_all_participated : total ≤ students_A + students_B) :
  students_in_both total students_A students_B = 25 := by
  sorry

#eval students_in_both 55 38 42  -- Should output 25

end students_in_both_competitions_l1416_141628


namespace regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l1416_141634

/-- The measure of an exterior angle of a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The measure of an exterior angle of a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle_is_45 : regular_octagon_exterior_angle = 45 := by
  sorry

end regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l1416_141634


namespace dog_food_consumption_l1416_141636

/-- The amount of dog food two dogs eat together daily -/
def total_food : ℝ := 0.25

/-- Given two dogs that eat the same amount of food daily, 
    prove that each dog eats half of the total food -/
theorem dog_food_consumption (dog1_food dog2_food : ℝ) 
  (h1 : dog1_food = dog2_food) 
  (h2 : dog1_food + dog2_food = total_food) : 
  dog1_food = 0.125 := by sorry

end dog_food_consumption_l1416_141636
