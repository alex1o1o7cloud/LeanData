import Mathlib

namespace exists_43_move_strategy_l2223_222301

/-- The number of boxes and chosen numbers -/
def n : ℕ := 2017

/-- A strategy for distributing stones -/
structure Strategy where
  numbers : Fin n → ℕ

/-- The state of the game after some moves -/
def GameState := Fin n → ℕ

/-- Apply a strategy for one move -/
def applyStrategy (s : Strategy) (state : GameState) : GameState :=
  fun i => state i + s.numbers i

/-- Apply a strategy for k moves -/
def applyStrategyKTimes (s : Strategy) (k : ℕ) : GameState :=
  fun i => k * (s.numbers i)

/-- Check if all boxes have the same number of stones -/
def allEqual (state : GameState) : Prop :=
  ∀ i j, state i = state j

/-- The main theorem -/
theorem exists_43_move_strategy :
  ∃ (s : Strategy),
    (allEqual (applyStrategyKTimes s 43)) ∧
    (∀ k, 0 < k → k < 43 → ¬(allEqual (applyStrategyKTimes s k))) := by
  sorry

end exists_43_move_strategy_l2223_222301


namespace distinct_residues_count_l2223_222369

theorem distinct_residues_count (n m : ℕ) (a b : ℕ → ℝ) :
  (∀ j ∈ Finset.range n, ∀ k ∈ Finset.range m, a j + b k ≠ 1) →
  (∀ j ∈ Finset.range (n-1), a j < a (j+1)) →
  (∀ k ∈ Finset.range (m-1), b k < b (k+1)) →
  a 0 = 0 →
  b 0 = 0 →
  (∀ j ∈ Finset.range n, 0 < a j ∧ a j < 1) →
  (∀ k ∈ Finset.range m, 0 < b k ∧ b k < 1) →
  Finset.card (Finset.image (λ (p : ℕ × ℕ) => (a p.1 + b p.2) % 1) (Finset.product (Finset.range n) (Finset.range m))) ≥ m + n - 1 :=
by sorry


end distinct_residues_count_l2223_222369


namespace shopkeeper_gain_l2223_222304

/-- Calculates the percentage gain for a shopkeeper given markup and discount percentages -/
theorem shopkeeper_gain (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  markup_percent = 35 →
  discount_percent = 20 →
  let marked_price := cost_price * (1 + markup_percent / 100)
  let selling_price := marked_price * (1 - discount_percent / 100)
  let gain := selling_price - cost_price
  gain / cost_price * 100 = 8 := by
  sorry

end shopkeeper_gain_l2223_222304


namespace scientific_notation_of_111_3_billion_l2223_222353

theorem scientific_notation_of_111_3_billion : ∃ (a : ℝ) (n : ℤ), 
  1 ≤ a ∧ a < 10 ∧ 111300000000 = a * (10 : ℝ) ^ n ∧ a = 1.113 ∧ n = 11 := by
  sorry

end scientific_notation_of_111_3_billion_l2223_222353


namespace two_digit_product_555_sum_l2223_222360

theorem two_digit_product_555_sum (x y : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 555 → x + y = 52 := by
  sorry

end two_digit_product_555_sum_l2223_222360


namespace quadratic_form_k_value_l2223_222347

theorem quadratic_form_k_value : 
  ∃ (a h k : ℚ), ∀ x, x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
  sorry

end quadratic_form_k_value_l2223_222347


namespace geometric_progression_equality_l2223_222381

/-- Given four real numbers a, b, c, d forming a geometric progression,
    prove that (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 -/
theorem geometric_progression_equality (a b c d : ℝ) 
  (h1 : c^2 = b * d) 
  (h2 : b^2 = a * c) 
  (h3 : a * d = b * c) : 
  (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 := by
  sorry

end geometric_progression_equality_l2223_222381


namespace derivative_x_ln_x_l2223_222363

open Real

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) :
  deriv (fun x => x * log x) x = log x + 1 := by
  sorry

end derivative_x_ln_x_l2223_222363


namespace total_choices_is_64_l2223_222339

/-- The number of tour routes available -/
def num_routes : ℕ := 4

/-- The number of tour groups -/
def num_groups : ℕ := 3

/-- The total number of different possible choices -/
def total_choices : ℕ := num_routes ^ num_groups

/-- Theorem stating that the total number of different choices is 64 -/
theorem total_choices_is_64 : total_choices = 64 := by
  sorry

end total_choices_is_64_l2223_222339


namespace unique_solution_l2223_222367

/-- The function F as defined in the problem -/
def F (t : ℝ) : ℝ := 32 * t^5 + 48 * t^3 + 17 * t - 15

/-- The system of equations -/
def system_equations (x y z : ℝ) : Prop :=
  1/x = 32/y^5 + 48/y^3 + 17/y - 15 ∧
  1/y = 32/z^5 + 48/z^3 + 17/z - 15 ∧
  1/z = 32/x^5 + 48/x^3 + 17/x - 15

/-- The theorem stating the unique solution -/
theorem unique_solution :
  ∃! (x y z : ℝ), system_equations x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end unique_solution_l2223_222367


namespace prime_divisor_of_3n_minus_1_and_n_minus_10_l2223_222371

theorem prime_divisor_of_3n_minus_1_and_n_minus_10 (n : ℕ) (p : ℕ) (h_prime : Prime p) 
  (h_div_3n_minus_1 : p ∣ (3 * n - 1)) (h_div_n_minus_10 : p ∣ (n - 10)) : p = 29 :=
sorry

end prime_divisor_of_3n_minus_1_and_n_minus_10_l2223_222371


namespace complement_A_intersect_B_l2223_222342

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 2, 5}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 5} := by sorry

end complement_A_intersect_B_l2223_222342


namespace chess_tournament_games_l2223_222324

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) :
  n = 12 →
  total_games = 132 →
  ∃ (games_per_pair : ℕ),
    total_games = games_per_pair * (n * (n - 1) / 2) ∧
    games_per_pair = 2 := by
  sorry

end chess_tournament_games_l2223_222324


namespace existence_of_xyz_l2223_222397

theorem existence_of_xyz (n : ℕ) (hn : n > 0) :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^(n-1) + y^n = z^(n+1) := by
  sorry

end existence_of_xyz_l2223_222397


namespace smaller_root_of_equation_l2223_222384

theorem smaller_root_of_equation (x : ℚ) :
  (x - 4/5)^2 + (x - 4/5) * (x - 2/5) + (x - 1/2)^2 = 0 →
  x = 14/15 ∨ x = 4/5 ∧ 14/15 < 4/5 := by
sorry

end smaller_root_of_equation_l2223_222384


namespace visible_cubes_count_l2223_222315

/-- Represents a cube with gaps -/
structure CubeWithGaps where
  size : ℕ
  unit_cubes : ℕ
  has_gaps : Bool

/-- Calculates the number of visible or partially visible unit cubes from a corner -/
def visible_cubes (c : CubeWithGaps) : ℕ :=
  sorry

/-- The specific cube in the problem -/
def problem_cube : CubeWithGaps :=
  { size := 12
  , unit_cubes := 12^3
  , has_gaps := true }

theorem visible_cubes_count :
  visible_cubes problem_cube = 412 :=
sorry

end visible_cubes_count_l2223_222315


namespace park_playgroups_l2223_222365

theorem park_playgroups (girls boys parents playgroups : ℕ) 
  (h1 : girls = 14)
  (h2 : boys = 11)
  (h3 : parents = 50)
  (h4 : playgroups = 3)
  (h5 : (girls + boys + parents) % playgroups = 0) :
  (girls + boys + parents) / playgroups = 25 := by
  sorry

end park_playgroups_l2223_222365


namespace triangle_angle_calculation_l2223_222329

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → a = Real.sqrt 3 → b = 1 →
  (0 < A ∧ A < π) → (0 < B ∧ B < π) → (0 < C ∧ C < π) →
  (A + B + C = π) →
  (Real.sin A / a = Real.sin B / b) →
  B = π/6 := by
sorry

end triangle_angle_calculation_l2223_222329


namespace quadratic_roots_imply_m_value_l2223_222352

theorem quadratic_roots_imply_m_value (m : ℝ) : 
  (∃ x : ℂ, 5 * x^2 - 4 * x + m = 0 ∧ 
   (x = (2 + Complex.I * Real.sqrt 143) / 5 ∨ 
    x = (2 - Complex.I * Real.sqrt 143) / 5)) → 
  m = 7.95 := by
sorry

end quadratic_roots_imply_m_value_l2223_222352


namespace total_pitchers_is_one_and_half_l2223_222372

/-- The total number of pitchers of lemonade served during a school play -/
def total_pitchers (first second third fourth : ℚ) : ℚ :=
  first + second + third + fourth

/-- Theorem stating that the total number of pitchers served is 1.5 -/
theorem total_pitchers_is_one_and_half :
  total_pitchers 0.25 0.4166666666666667 0.25 0.5833333333333334 = 1.5 := by
  sorry

end total_pitchers_is_one_and_half_l2223_222372


namespace volume_ratio_cylinders_capacity_ratio_64_percent_l2223_222335

/-- The volume ratio of two right circular cylinders with the same height
    is equal to the square of the ratio of their circumferences. -/
theorem volume_ratio_cylinders (h C_A C_B : ℝ) (h_pos : h > 0) (C_A_pos : C_A > 0) (C_B_pos : C_B > 0) :
  (h * (C_A / (2 * Real.pi))^2) / (h * (C_B / (2 * Real.pi))^2) = (C_A / C_B)^2 := by
  sorry

/-- The capacity of a cylinder with circumference 8 is 64% of the capacity
    of a cylinder with circumference 10, given the same height. -/
theorem capacity_ratio_64_percent (h : ℝ) (h_pos : h > 0) :
  (h * (8 / (2 * Real.pi))^2) / (h * (10 / (2 * Real.pi))^2) = 0.64 := by
  sorry

end volume_ratio_cylinders_capacity_ratio_64_percent_l2223_222335


namespace cube_sum_equation_l2223_222343

theorem cube_sum_equation (a b : ℝ) 
  (h1 : a^5 - a^4*b - a^4 + a - b - 1 = 0)
  (h2 : 2*a - 3*b = 1) : 
  a^3 + b^3 = 9 := by sorry

end cube_sum_equation_l2223_222343


namespace fourth_column_is_quadratic_l2223_222309

/-- A quadruple of real numbers is quadratic if it satisfies the quadratic condition. -/
def is_quadratic (y : Fin 4 → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ n : Fin 4, y n = a * (n.val + 1)^2 + b * (n.val + 1) + c

/-- A 4×4 grid of real numbers. -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- All rows of the grid are quadratic. -/
def all_rows_quadratic (g : Grid) : Prop :=
  ∀ i : Fin 4, is_quadratic (λ j => g i j)

/-- The first three columns of the grid are quadratic. -/
def first_three_columns_quadratic (g : Grid) : Prop :=
  ∀ j : Fin 3, is_quadratic (λ i => g i j)

/-- The fourth column of the grid is quadratic. -/
def fourth_column_quadratic (g : Grid) : Prop :=
  is_quadratic (λ i => g i 3)

/-- 
If all rows and the first three columns of a 4×4 grid are quadratic,
then the fourth column is also quadratic.
-/
theorem fourth_column_is_quadratic (g : Grid)
  (h_rows : all_rows_quadratic g)
  (h_cols : first_three_columns_quadratic g) :
  fourth_column_quadratic g :=
sorry

end fourth_column_is_quadratic_l2223_222309


namespace vacation_pictures_l2223_222326

def remaining_pictures (zoo_pics : ℕ) (museum_pics : ℕ) (deleted_pics : ℕ) : ℕ :=
  zoo_pics + museum_pics - deleted_pics

theorem vacation_pictures (zoo_pics : ℕ) (museum_pics : ℕ) (deleted_pics : ℕ) 
  (h1 : zoo_pics = 50)
  (h2 : museum_pics = 8)
  (h3 : deleted_pics = 38) :
  remaining_pictures zoo_pics museum_pics deleted_pics = 20 := by
  sorry

end vacation_pictures_l2223_222326


namespace complex_magnitude_problem_l2223_222393

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z + w) = 6) :
  Complex.abs z = Real.sqrt 15 := by
  sorry

end complex_magnitude_problem_l2223_222393


namespace painter_problem_l2223_222341

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculates the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Proves that for the given scenario, the time to paint the remaining rooms is 32 hours. -/
theorem painter_problem :
  let total_rooms : ℕ := 9
  let time_per_room : ℕ := 8
  let painted_rooms : ℕ := 5
  time_to_paint_remaining total_rooms time_per_room painted_rooms = 32 :=
by
  sorry


end painter_problem_l2223_222341


namespace point_transformation_l2223_222340

/-- Given a point P(a,b) in the xy-plane, this theorem proves that if P is first rotated
    clockwise by 180° around the origin (0,0) and then reflected about the line y = -x,
    resulting in the point (9,-4), then b - a = -13. -/
theorem point_transformation (a b : ℝ) : 
  (∃ (x y : ℝ), ((-a) = x ∧ (-b) = y) ∧ (y = x ∧ -x = 9 ∧ -y = -4)) → b - a = -13 := by
  sorry

end point_transformation_l2223_222340


namespace only_crop_yield_fertilizer_correlational_l2223_222345

-- Define the types of relationships
inductive Relationship
| Functional
| Correlational

-- Define the variables for each relationship
def height_age_relation : Relationship := sorry
def cube_volume_edge_relation : Relationship := sorry
def pencils_money_relation : Relationship := sorry
def crop_yield_fertilizer_relation : Relationship := sorry

-- Theorem stating that only the crop yield and fertilizer relationship is correlational
theorem only_crop_yield_fertilizer_correlational :
  (height_age_relation = Relationship.Functional) ∧
  (cube_volume_edge_relation = Relationship.Functional) ∧
  (pencils_money_relation = Relationship.Functional) ∧
  (crop_yield_fertilizer_relation = Relationship.Correlational) := by sorry

end only_crop_yield_fertilizer_correlational_l2223_222345


namespace parabola_circle_tangency_l2223_222389

/-- The value of m for which the line x = -2 is tangent to the circle x^2 + y^2 + 6x + m = 0 -/
theorem parabola_circle_tangency (m : ℝ) : 
  (∀ y : ℝ, ((-2)^2 + y^2 + 6*(-2) + m = 0) → 
   (∀ x : ℝ, x ≠ -2 → x^2 + y^2 + 6*x + m ≠ 0)) → 
  m = 8 := by sorry

end parabola_circle_tangency_l2223_222389


namespace locus_of_perpendicular_foot_l2223_222305

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a rectangular hyperbola -/
structure RectangularHyperbola where
  k : ℝ

/-- Represents the locus of points -/
def locus (h : RectangularHyperbola) : Set PolarPoint :=
  {p : PolarPoint | p.r^2 = 2 * h.k^2 * Real.sin (2 * p.θ)}

/-- The main theorem stating that the locus of the foot of the perpendicular
    from the center of a rectangular hyperbola to a tangent is given by
    the polar equation r^2 = 2k^2 sin 2θ -/
theorem locus_of_perpendicular_foot (h : RectangularHyperbola) :
  ∀ p : PolarPoint, p ∈ locus h ↔
    ∃ (t : ℝ), -- t represents the parameter of a point on the hyperbola
      let tangent_point := (t, h.k^2 / t)
      let tangent_slope := -h.k^2 / t^2
      let perpendicular_slope := -1 / tangent_slope
      p.r * (Real.cos p.θ) = 2 * h.k^4 / (t * (t^4 + h.k^4)) ∧
      p.r * (Real.sin p.θ) = 2 * t * h.k^2 / (t^4 + h.k^4) :=
by
  sorry

end locus_of_perpendicular_foot_l2223_222305


namespace exists_term_with_four_zero_digits_l2223_222319

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem exists_term_with_four_zero_digits : 
  ∃ n : ℕ, n < 100000001 ∧ last_four_digits (fibonacci n) = 0 :=
sorry

end exists_term_with_four_zero_digits_l2223_222319


namespace max_a_value_l2223_222377

theorem max_a_value (a b c d : ℕ+) 
  (h1 : a < 2 * b + 1)
  (h2 : b < 3 * c + 1)
  (h3 : c < 4 * d + 1)
  (h4 : d^2 < 10000) :
  a ≤ 2376 :=
by sorry

end max_a_value_l2223_222377


namespace smallest_solution_congruence_l2223_222322

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 33 = 14 % 33 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 33 = 14 % 33 → x ≤ y :=
by
  -- The proof would go here
  sorry

end smallest_solution_congruence_l2223_222322


namespace f_odd_and_increasing_l2223_222338

def f (x : ℝ) := x^3 + x

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end f_odd_and_increasing_l2223_222338


namespace film_festival_selection_l2223_222394

/-- Given a film festival selection process, prove that the fraction of color films
    selected by the subcommittee is 20/21. -/
theorem film_festival_selection (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let total_films := 30 * x + 6 * y
  let bw_selected := (y / x) * (30 * x) / 100
  let color_selected := 6 * y
  let total_selected := bw_selected + color_selected
  color_selected / total_selected = 20 / 21 := by
  sorry

end film_festival_selection_l2223_222394


namespace correct_ages_l2223_222337

/-- Represents the ages of family members -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  mother : ℕ

/-- Calculates the correct ages given the problem conditions -/
def calculateAges : FamilyAges :=
  let father := 44
  let son := father / 2
  let mother := son + 5
  { father := father, son := son, mother := mother }

/-- Theorem stating that the calculated ages satisfy the given conditions -/
theorem correct_ages (ages : FamilyAges := calculateAges) :
  ages.father = 44 ∧
  ages.father = ages.son + ages.son ∧
  ages.son - 5 = ages.mother - 10 ∧
  ages.father = 44 ∧
  ages.son = 22 ∧
  ages.mother = 27 :=
by sorry

end correct_ages_l2223_222337


namespace max_triangle_area_l2223_222313

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def isosceles_trapezoid (m n f2 f1 : Point) : Prop :=
  ∃ (height area : ℝ), height = Real.sqrt 3 ∧ area = 3 * Real.sqrt 3

def line_through_point (p : Point) : Set Point :=
  {q : Point | ∃ (k : ℝ), q.x = k * q.y + p.x}

def intersect_ellipse_line (e : Ellipse) (l : Set Point) : Set Point :=
  {p : Point | p ∈ l ∧ e.equation p}

def triangle_area (t : Triangle) : ℝ :=
  sorry

theorem max_triangle_area (e : Ellipse) (m n f2 f1 : Point) :
  isosceles_trapezoid m n f2 f1 →
  m = Point.mk (-e.a) e.b →
  n = Point.mk e.a e.b →
  (∀ (l : Set Point), f1 ∈ l →
    let intersection := intersect_ellipse_line e l
    ∀ (a b : Point), a ∈ intersection → b ∈ intersection →
      triangle_area (Triangle.mk f2 a b) ≤ 3) ∧
  (∃ (l : Set Point), f1 ∈ l →
    let intersection := intersect_ellipse_line e l
    ∃ (a b : Point), a ∈ intersection ∧ b ∈ intersection ∧
      triangle_area (Triangle.mk f2 a b) = 3) :=
sorry

end max_triangle_area_l2223_222313


namespace cricket_team_captain_age_l2223_222395

theorem cricket_team_captain_age (team_size : ℕ) (captain_age wicket_keeper_age : ℕ) 
  (team_average : ℚ) (remaining_average : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 3 →
  team_average = 22 →
  remaining_average = team_average - 1 →
  (team_size : ℚ) * team_average = 
    captain_age + wicket_keeper_age + (team_size - 2 : ℚ) * remaining_average →
  captain_age = 25 := by
sorry

end cricket_team_captain_age_l2223_222395


namespace correct_average_marks_l2223_222328

theorem correct_average_marks (n : ℕ) (initial_avg : ℚ) (wrong_mark correct_mark : ℚ) :
  n = 30 →
  initial_avg = 100 →
  wrong_mark = 70 →
  correct_mark = 10 →
  (n * initial_avg - (wrong_mark - correct_mark)) / n = 98 := by
  sorry

end correct_average_marks_l2223_222328


namespace f_properties_l2223_222373

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 2 * Real.sin x

theorem f_properties :
  (∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ max = Real.sqrt 3) ∧
  (∃ (θ : ℝ), ∀ (x : ℝ), f x ≤ f θ ∧ Real.cos (θ - π/6) = (3 + Real.sqrt 6) / 6) :=
sorry

end f_properties_l2223_222373


namespace same_number_of_heads_probability_p_plus_q_l2223_222311

-- Define the probability of heads for a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the probability of heads for the biased coin
def biased_coin_prob : ℚ := 2/5

-- Define the function to calculate the probability of getting k heads when flipping both coins
def prob_k_heads (k : ℕ) : ℚ :=
  match k with
  | 0 => (1 - fair_coin_prob) * (1 - biased_coin_prob)
  | 1 => fair_coin_prob * (1 - biased_coin_prob) + (1 - fair_coin_prob) * biased_coin_prob
  | 2 => fair_coin_prob * biased_coin_prob
  | _ => 0

-- State the theorem
theorem same_number_of_heads_probability :
  (prob_k_heads 0)^2 + (prob_k_heads 1)^2 + (prob_k_heads 2)^2 = 19/50 := by
  sorry

-- Define p and q
def p : ℕ := 19
def q : ℕ := 50

-- State the theorem for p + q
theorem p_plus_q : p + q = 69 := by
  sorry

end same_number_of_heads_probability_p_plus_q_l2223_222311


namespace quadratic_equation_condition_l2223_222366

/-- For the equation (a-2)x^2 + (a+2)x + 3 = 0 to be a quadratic equation in one variable, a ≠ 2 -/
theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, ∃ y, y = (a - 2) * x^2 + (a + 2) * x + 3) → a ≠ 2 := by
  sorry

end quadratic_equation_condition_l2223_222366


namespace quotient_sum_difference_forty_percent_less_than_36_l2223_222375

-- Problem 1
theorem quotient_sum_difference : (0.4 + 1/3) / (0.4 - 1/3) = 11 := by sorry

-- Problem 2
theorem forty_percent_less_than_36 : ∃ x : ℝ, x - 0.4 * x = 36 ∧ x = 60 := by sorry

end quotient_sum_difference_forty_percent_less_than_36_l2223_222375


namespace vector_magnitude_problem_l2223_222310

/-- Given two vectors in 2D space satisfying certain conditions, prove that the magnitude of one vector is 3. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  (∃ θ : ℝ, θ = Real.pi / 3 ∧ a.fst * b.fst + a.snd * b.snd = Real.cos θ * ‖a‖ * ‖b‖) →  -- angle between a and b is 60°
  ‖a‖ = 1 →  -- |a| = 1
  ‖2 • a - b‖ = Real.sqrt 7 →  -- |2a - b| = √7
  ‖b‖ = 3 := by
sorry


end vector_magnitude_problem_l2223_222310


namespace quadratic_always_positive_implies_m_greater_than_one_l2223_222387

theorem quadratic_always_positive_implies_m_greater_than_one (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + m > 0) → m > 1 := by
  sorry

end quadratic_always_positive_implies_m_greater_than_one_l2223_222387


namespace salad_dressing_calories_l2223_222312

/-- Calculates the calories in the salad dressing given the total calories consumed and the calories from other ingredients. -/
theorem salad_dressing_calories :
  let lettuce_calories : ℝ := 50
  let carrot_calories : ℝ := 2 * lettuce_calories
  let pizza_crust_calories : ℝ := 600
  let pepperoni_calories : ℝ := (1 / 3) * pizza_crust_calories
  let cheese_calories : ℝ := 400
  let salad_portion : ℝ := 1 / 4
  let pizza_portion : ℝ := 1 / 5
  let total_calories_consumed : ℝ := 330

  let salad_calories_without_dressing : ℝ := (lettuce_calories + carrot_calories) * salad_portion
  let pizza_calories : ℝ := (pizza_crust_calories + pepperoni_calories + cheese_calories) * pizza_portion
  let calories_without_dressing : ℝ := salad_calories_without_dressing + pizza_calories
  let dressing_calories : ℝ := total_calories_consumed - calories_without_dressing

  dressing_calories = 52.5 := by sorry

end salad_dressing_calories_l2223_222312


namespace evaluate_nested_fraction_l2223_222385

theorem evaluate_nested_fraction : 1 - (1 / (1 - (1 / (1 + 2)))) = -1 / 2 := by sorry

end evaluate_nested_fraction_l2223_222385


namespace student_A_pass_probability_l2223_222323

/-- Probability that student A passes the exam --/
def prob_pass (pA pB pDAC pDB : ℝ) : ℝ :=
  pA * pDAC + pB * pDB + (1 - pA - pB) * pDAC

theorem student_A_pass_probability :
  let pA := 0.3
  let pB := 0.3
  let pDAC := 0.8
  let pDB := 0.6
  prob_pass pA pB pDAC pDB = 0.74 := by
  sorry

#eval prob_pass 0.3 0.3 0.8 0.6

end student_A_pass_probability_l2223_222323


namespace intersection_condition_l2223_222383

/-- Curve C in the xy-plane -/
def C (x y : ℝ) : Prop := y^2 = 6*x - 2 ∧ y ≥ 0

/-- Line l in the xy-plane -/
def L (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + 2*m

/-- Intersection points of C and L -/
def Intersection (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C p.1 p.2 ∧ L p.1 p.2 m}

/-- Two distinct intersection points exist -/
def HasTwoDistinctIntersections (m : ℝ) : Prop :=
  ∃ p q : ℝ × ℝ, p ∈ Intersection m ∧ q ∈ Intersection m ∧ p ≠ q

theorem intersection_condition (m : ℝ) :
  HasTwoDistinctIntersections m ↔ -Real.sqrt 3 / 6 ≤ m ∧ m < Real.sqrt 3 / 12 := by
  sorry

end intersection_condition_l2223_222383


namespace travel_distance_l2223_222358

theorem travel_distance (d : ℝ) (h1 : d > 0) :
  d / 4 + d / 8 + d / 12 = 11 / 60 →
  3 * d = 1.2 := by
sorry

end travel_distance_l2223_222358


namespace question_selection_ways_l2223_222356

theorem question_selection_ways : 
  (Nat.choose 10 8) * (Nat.choose 10 5) = 11340 := by sorry

end question_selection_ways_l2223_222356


namespace cuts_through_examples_l2223_222344

-- Define what it means for a line to cut through a curve at a point
def cuts_through (l : ℝ → ℝ) (c : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  -- The line is tangent to the curve at the point
  (∀ x, l x = c x + (l p.1 - c p.1)) ∧
  -- The curve is on both sides of the line near the point
  ∃ δ > 0, ∀ x ∈ Set.Ioo (p.1 - δ) (p.1 + δ), 
    (x < p.1 → c x < l x) ∧ (x > p.1 → c x > l x)

-- Theorem statement
theorem cuts_through_examples :
  cuts_through (λ _ => 0) (λ x => x^3) (0, 0) ∧
  cuts_through (λ x => x) Real.sin (0, 0) ∧
  cuts_through (λ x => x) Real.tan (0, 0) := by
  sorry

end cuts_through_examples_l2223_222344


namespace rays_grocery_bill_l2223_222349

/-- Calculates the total grocery bill for Ray's purchase with a rewards discount --/
theorem rays_grocery_bill :
  let hamburger_price : ℚ := 5
  let crackers_price : ℚ := 3.5
  let vegetable_price : ℚ := 2
  let vegetable_quantity : ℕ := 4
  let cheese_price : ℚ := 3.5
  let discount_rate : ℚ := 0.1

  let total_before_discount : ℚ := 
    hamburger_price + crackers_price + (vegetable_price * vegetable_quantity) + cheese_price
  
  let discount_amount : ℚ := total_before_discount * discount_rate
  
  let final_bill : ℚ := total_before_discount - discount_amount

  final_bill = 18 := by sorry

end rays_grocery_bill_l2223_222349


namespace dot_product_specific_vectors_l2223_222300

theorem dot_product_specific_vectors :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-3, 1)
  (a.1 * b.1 + a.2 * b.2) = -1 := by sorry

end dot_product_specific_vectors_l2223_222300


namespace next_two_pythagorean_triples_l2223_222370

/-- Given a sequence of Pythagorean triples, find the next two triples -/
theorem next_two_pythagorean_triples 
  (h1 : 3^2 + 4^2 = 5^2)
  (h2 : 5^2 + 12^2 = 13^2)
  (h3 : 7^2 + 24^2 = 25^2) :
  (9^2 + 40^2 = 41^2) ∧ (11^2 + 60^2 = 61^2) := by
  sorry

end next_two_pythagorean_triples_l2223_222370


namespace simplified_expression_equals_negative_sqrt_three_l2223_222390

theorem simplified_expression_equals_negative_sqrt_three :
  let a := 2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180)
  let b := 3
  1 - (a - b) / (a + 2 * b) / ((a^2 - b^2) / (a^2 + 4 * a * b + 4 * b^2)) = -Real.sqrt 3 := by
  sorry

end simplified_expression_equals_negative_sqrt_three_l2223_222390


namespace second_digit_of_n_l2223_222378

theorem second_digit_of_n (n : ℕ) : 
  (10^99 ≤ 8*n) ∧ (8*n < 10^100) ∧ 
  (10^101 ≤ 81*n - 102) ∧ (81*n - 102 < 10^102) →
  (n / 10^97) % 10 = 2 :=
by sorry

end second_digit_of_n_l2223_222378


namespace twenty_thousand_scientific_notation_l2223_222379

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  prop : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_thousand_scientific_notation :
  toScientificNotation 20000 = ScientificNotation.mk 2 4 (by norm_num) :=
sorry

end twenty_thousand_scientific_notation_l2223_222379


namespace worker_earnings_worker_earnings_proof_l2223_222332

/-- Calculates the total earnings of a worker based on regular and cellphone survey rates -/
theorem worker_earnings (regular_rate : ℕ) (total_surveys : ℕ) (cellphone_rate_increase : ℚ) 
  (cellphone_surveys : ℕ) (h1 : regular_rate = 30) (h2 : total_surveys = 100) 
  (h3 : cellphone_rate_increase = 1/5) (h4 : cellphone_surveys = 50) : ℕ :=
  let regular_surveys := total_surveys - cellphone_surveys
  let cellphone_rate := regular_rate + (regular_rate * cellphone_rate_increase).floor
  let regular_pay := regular_surveys * regular_rate
  let cellphone_pay := cellphone_surveys * cellphone_rate
  let total_pay := regular_pay + cellphone_pay
  3300

/-- The worker's total earnings for the week are Rs. 3300 -/
theorem worker_earnings_proof : worker_earnings 30 100 (1/5) 50 rfl rfl rfl rfl = 3300 := by
  sorry

end worker_earnings_worker_earnings_proof_l2223_222332


namespace quadratic_root_range_l2223_222399

theorem quadratic_root_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 3) ↔ x ≥ -3 :=
by sorry

end quadratic_root_range_l2223_222399


namespace negation_of_universal_quantifier_negation_of_inequality_negation_of_proposition_l2223_222318

theorem negation_of_universal_quantifier (P : ℝ → Prop) :
  (¬ ∀ x ∈ Set.Ioo 0 1, P x) ↔ (∃ x ∈ Set.Ioo 0 1, ¬ P x) := by sorry

theorem negation_of_inequality (x : ℝ) : ¬(x^2 - x < 0) ↔ x^2 - x ≥ 0 := by sorry

theorem negation_of_proposition :
  (¬ ∀ x ∈ Set.Ioo 0 1, x^2 - x < 0) ↔ (∃ x ∈ Set.Ioo 0 1, x^2 - x ≥ 0) := by sorry

end negation_of_universal_quantifier_negation_of_inequality_negation_of_proposition_l2223_222318


namespace box_dimensions_l2223_222302

theorem box_dimensions (a h : ℝ) (b : ℝ) : 
  h = a / 2 →
  6 * a + b = 156 →
  7 * a + b = 178 →
  a = 22 ∧ h = 11 :=
by sorry

end box_dimensions_l2223_222302


namespace remainder_theorem_l2223_222351

-- Define the polynomial p(x) = 4x^3 - 12x^2 + 16x - 20
def p (x : ℝ) : ℝ := 4 * x^3 - 12 * x^2 + 16 * x - 20

-- Define the divisor d(x) = x - 3
def d (x : ℝ) : ℝ := x - 3

-- Theorem statement
theorem remainder_theorem :
  (p 3 : ℝ) = 28 := by sorry

end remainder_theorem_l2223_222351


namespace opposite_sqrt_nine_is_negative_three_l2223_222331

theorem opposite_sqrt_nine_is_negative_three :
  -(Real.sqrt 9) = -3 := by
  sorry

end opposite_sqrt_nine_is_negative_three_l2223_222331


namespace P_and_S_not_third_l2223_222386

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the finish order relation
def finishes_before (a b : Runner) : Prop := sorry

-- Define the race conditions
axiom P_beats_Q : finishes_before Runner.P Runner.Q
axiom P_beats_R : finishes_before Runner.P Runner.R
axiom Q_beats_S : finishes_before Runner.Q Runner.S
axiom U_after_P_before_T : finishes_before Runner.P Runner.U ∧ finishes_before Runner.U Runner.T
axiom T_after_P_before_Q : finishes_before Runner.P Runner.T ∧ finishes_before Runner.T Runner.Q

-- Define a function to represent the finishing position of a runner
def finish_position (r : Runner) : ℕ := sorry

-- State the theorem
theorem P_and_S_not_third :
  ¬(finish_position Runner.P = 3 ∨ finish_position Runner.S = 3) :=
sorry

end P_and_S_not_third_l2223_222386


namespace expression_evaluation_l2223_222321

theorem expression_evaluation : 2^3 + 15 * 2 - 4 + 10 * 5 / 2 = 59 := by
  sorry

end expression_evaluation_l2223_222321


namespace average_weight_increase_l2223_222346

/-- Proves that replacing a person weighing 45 kg with a person weighing 65 kg
    in a group of 8 people increases the average weight by 2.5 kg -/
theorem average_weight_increase (initial_group_size : ℕ) 
                                 (old_weight new_weight : ℝ) : 
  initial_group_size = 8 →
  old_weight = 45 →
  new_weight = 65 →
  (new_weight - old_weight) / initial_group_size = 2.5 := by
  sorry

end average_weight_increase_l2223_222346


namespace inequality_system_solution_l2223_222374

theorem inequality_system_solution :
  ∃ (x : ℤ),
    (3 * (2 * x - 1) < 2 * x + 8) ∧
    (2 + (3 * (x + 1)) / 8 > 3 - (x - 1) / 4) ∧
    (x = 2) ∧
    (∀ a : ℝ, (a * x + 6 ≤ x - 2 * a) → (|a + 1| - |a - 1| = -2)) :=
by sorry

end inequality_system_solution_l2223_222374


namespace barn_painted_area_l2223_222388

/-- Calculates the total area to be painted for a rectangular barn --/
def total_painted_area (width length height : ℝ) : ℝ :=
  let wall_area := 2 * (width * height + length * height)
  let floor_ceiling_area := 2 * (width * length)
  wall_area + floor_ceiling_area

/-- Theorem stating that the total area to be painted for the given barn is 1002 sq yd --/
theorem barn_painted_area :
  total_painted_area 15 18 7 = 1002 := by
  sorry

end barn_painted_area_l2223_222388


namespace class_artworks_l2223_222368

theorem class_artworks (total_students : ℕ) (total_kits : ℕ) 
  (students_one_kit : ℕ) (students_two_kits : ℕ)
  (students_five_works : ℕ) (students_six_works : ℕ) (students_seven_works : ℕ) :
  total_students = 24 →
  total_kits = 36 →
  students_one_kit = 12 →
  students_two_kits = 12 →
  students_five_works = 8 →
  students_six_works = 10 →
  students_seven_works = 6 →
  students_one_kit + students_two_kits = total_students →
  students_five_works + students_six_works + students_seven_works = total_students →
  students_five_works * 5 + students_six_works * 6 + students_seven_works * 7 = 142 :=
by sorry

end class_artworks_l2223_222368


namespace point_on_inverse_proportion_graph_l2223_222333

theorem point_on_inverse_proportion_graph :
  let f : ℝ → ℝ := λ x => 6 / x
  f 2 = 3 :=
by
  sorry

end point_on_inverse_proportion_graph_l2223_222333


namespace parabola_intersection_value_l2223_222362

theorem parabola_intersection_value (a : ℝ) : 
  a^2 - a - 1 = 0 → a^2 - a + 2014 = 2015 := by
  sorry

end parabola_intersection_value_l2223_222362


namespace triangle_area_in_square_grid_l2223_222316

theorem triangle_area_in_square_grid :
  let square_side : ℝ := 4
  let square_area : ℝ := square_side ^ 2
  let triangle1_area : ℝ := 4
  let triangle2_area : ℝ := 2
  let triangle3_area : ℝ := 3
  let total_triangles_area : ℝ := triangle1_area + triangle2_area + triangle3_area
  let triangle_abc_area : ℝ := square_area - total_triangles_area
  triangle_abc_area = 7 := by sorry

end triangle_area_in_square_grid_l2223_222316


namespace total_shoes_count_l2223_222303

theorem total_shoes_count (bonny becky bobby cherry diane : ℕ) : 
  bonny = 13 ∧
  bonny = 2 * becky - 5 ∧
  bobby = 3 * becky ∧
  cherry = bonny + becky + 4 ∧
  diane = 2 * cherry - 2 →
  bonny + becky + bobby + cherry + diane = 125 := by
  sorry

end total_shoes_count_l2223_222303


namespace max_expressions_greater_than_one_two_expressions_can_be_greater_than_one_max_expressions_greater_than_one_is_two_l2223_222317

theorem max_expressions_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  (∃ (x y z : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧ z ∈ Set.range p ∧
    x > 1 ∧ y > 1 ∧ z > 1) → False :=
by sorry

theorem two_expressions_can_be_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ∃ (x y : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧
    x > 1 ∧ y > 1 :=
by sorry

theorem max_expressions_greater_than_one_is_two (a b c : ℝ) (h : a * b * c = 1) :
  (∃ (x y : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧
    x > 1 ∧ y > 1) ∧
  (∃ (x y z : ℝ) (p : Fin 3 → ℝ), 
    (p 0 = 2*a - 1/b) ∧ (p 1 = 2*b - 1/c) ∧ (p 2 = 2*c - 1/a) ∧
    x ∈ Set.range p ∧ y ∈ Set.range p ∧ z ∈ Set.range p ∧
    x > 1 ∧ y > 1 ∧ z > 1) → False :=
by sorry

end max_expressions_greater_than_one_two_expressions_can_be_greater_than_one_max_expressions_greater_than_one_is_two_l2223_222317


namespace jason_total_cost_l2223_222380

def stove_cost : ℚ := 1200
def wall_cost : ℚ := stove_cost / 6
def repair_cost : ℚ := stove_cost + wall_cost
def labor_fee_rate : ℚ := 1/5  -- 20% as a fraction

def total_cost : ℚ := repair_cost + (labor_fee_rate * repair_cost)

theorem jason_total_cost : total_cost = 1680 := by
  sorry

end jason_total_cost_l2223_222380


namespace triangle_side_length_l2223_222350

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 9 → b = 2 * Real.sqrt 3 → C = 150 * π / 180 → c = 7 * Real.sqrt 3 := by
  sorry

end triangle_side_length_l2223_222350


namespace angle_conversion_l2223_222348

theorem angle_conversion (α k : ℤ) : 
  α = 195 ∧ k = -3 → 
  0 ≤ α ∧ α < 360 ∧ 
  -885 = α + k * 360 := by
  sorry

end angle_conversion_l2223_222348


namespace problems_solved_l2223_222327

theorem problems_solved (first last : ℕ) (h : first = 55) (h' : last = 150) :
  (Finset.range (last - first + 1)).card = 96 := by
  sorry

end problems_solved_l2223_222327


namespace logarithm_expression_equals_three_l2223_222376

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_three :
  log10 5^2 + 2/3 * log10 8 + log10 5 * log10 20 + (log10 2)^2 = 3 := by
  sorry

end logarithm_expression_equals_three_l2223_222376


namespace tomato_ratio_l2223_222336

def total_tomatoes : ℕ := 127
def eaten_by_birds : ℕ := 19
def tomatoes_left : ℕ := 54

theorem tomato_ratio :
  let picked := total_tomatoes - eaten_by_birds
  let given_to_friend := picked - tomatoes_left
  (given_to_friend : ℚ) / picked = 1 / 2 := by
  sorry

end tomato_ratio_l2223_222336


namespace sum_of_cubes_l2223_222359

theorem sum_of_cubes (a b s p : ℝ) (h1 : s = a + b) (h2 : p = a * b) : 
  a^3 + b^3 = s^3 - 3*s*p := by sorry

end sum_of_cubes_l2223_222359


namespace second_number_proof_l2223_222396

theorem second_number_proof (first second third : ℝ) : 
  first = 6 → 
  third = 22 → 
  (first + second + third) / 3 = 13 → 
  second = 11 := by
sorry

end second_number_proof_l2223_222396


namespace solve_for_b_l2223_222398

theorem solve_for_b (a b : ℝ) (h1 : 3 * a + 2 = 5) (h2 : b - 4 * a = 2) : b = 6 := by
  sorry

end solve_for_b_l2223_222398


namespace perimeter_specific_midpoint_triangle_l2223_222361

/-- A solid right prism with regular hexagonal bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Midpoint of an edge -/
structure Midpoint where
  edge : String

/-- Triangle formed by three midpoints -/
structure MidpointTriangle where
  point1 : Midpoint
  point2 : Midpoint
  point3 : Midpoint

/-- Calculate the perimeter of the midpoint triangle -/
def perimeter_midpoint_triangle (prism : RightPrism) (triangle : MidpointTriangle) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific midpoint triangle -/
theorem perimeter_specific_midpoint_triangle :
  ∀ (prism : RightPrism) (triangle : MidpointTriangle),
  prism.height = 20 ∧ 
  prism.base_side_length = 10 ∧
  triangle.point1 = Midpoint.mk "AB" ∧
  triangle.point2 = Midpoint.mk "BC" ∧
  triangle.point3 = Midpoint.mk "EF" →
  perimeter_midpoint_triangle prism triangle = 45 :=
sorry

end perimeter_specific_midpoint_triangle_l2223_222361


namespace second_number_proof_l2223_222320

theorem second_number_proof (x : ℕ) : x > 1428 ∧ 
  x % 129 = 13 ∧ 
  1428 % 129 = 9 ∧ 
  (∀ y, y > 1428 ∧ y % 129 = 13 → y ≥ x) ∧ 
  x = 1561 := by
sorry

end second_number_proof_l2223_222320


namespace zeros_difference_quadratic_l2223_222308

theorem zeros_difference_quadratic (m : ℝ) : 
  (∃ α β : ℝ, 2 * α^2 - m * α - 8 = 0 ∧ 
              2 * β^2 - m * β - 8 = 0 ∧ 
              α - β = m - 1) ↔ 
  (m = 6 ∨ m = -10/3) := by
sorry

end zeros_difference_quadratic_l2223_222308


namespace intersection_of_A_and_B_l2223_222307

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l2223_222307


namespace circle_condition_l2223_222391

theorem circle_condition (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4*k*x - 2*y + 5*k = 0 ∧ 
   ∀ (x' y' : ℝ), x'^2 + y'^2 + 4*k*x' - 2*y' + 5*k = 0 → 
   (x' - x)^2 + (y' - y)^2 = (x - x)^2 + (y - y)^2) ↔ 
  (k > 1 ∨ k < 1/4) :=
sorry

end circle_condition_l2223_222391


namespace percentage_calculation_l2223_222330

theorem percentage_calculation (P : ℝ) : 
  (P / 100) * 1265 / 5.96 = 377.8020134228188 → P = 178 := by
  sorry

end percentage_calculation_l2223_222330


namespace prime_triplet_divisibility_l2223_222355

theorem prime_triplet_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧
  (q * r - 1) % p = 0 ∧
  (p * r - 1) % q = 0 ∧
  (p * q - 1) % r = 0 →
  ({p, q, r} : Set ℕ) = {2, 3, 5} :=
by sorry

end prime_triplet_divisibility_l2223_222355


namespace chessboard_polygon_theorem_l2223_222364

/-- A polygon cut out from an infinite chessboard -/
structure ChessboardPolygon where
  black_cells : ℕ              -- number of black cells
  white_cells : ℕ              -- number of white cells
  black_perimeter : ℕ          -- number of black perimeter segments
  white_perimeter : ℕ          -- number of white perimeter segments

/-- Theorem stating the relationship between perimeter segments and cells -/
theorem chessboard_polygon_theorem (p : ChessboardPolygon) :
  p.black_perimeter - p.white_perimeter = 4 * (p.black_cells - p.white_cells) := by
  sorry

end chessboard_polygon_theorem_l2223_222364


namespace remainder_problem_l2223_222334

theorem remainder_problem (n : ℤ) (h : n % 5 = 3) : (n + 1) % 5 = 4 := by
  sorry

end remainder_problem_l2223_222334


namespace function_composition_l2223_222325

theorem function_composition (f : ℝ → ℝ) (x : ℝ) :
  (∀ y, f y = y^2 + 2*y) →
  f (2*x + 1) = 4*x^2 + 8*x + 3 := by
sorry

end function_composition_l2223_222325


namespace brick_length_calculation_l2223_222354

theorem brick_length_calculation (courtyard_length courtyard_width : ℝ)
  (brick_width : ℝ) (total_bricks : ℕ) (h1 : courtyard_length = 25)
  (h2 : courtyard_width = 16) (h3 : brick_width = 0.1) (h4 : total_bricks = 20000) :
  (courtyard_length * courtyard_width * 10000) / (brick_width * total_bricks) = 20 := by
  sorry

end brick_length_calculation_l2223_222354


namespace triangle_area_l2223_222306

/-- The area of the right triangle formed by the x-axis, the line y = 2, and the line x = 1 + √3y --/
theorem triangle_area : ℝ := by
  -- Define the lines
  let x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}
  let line_y2 : Set (ℝ × ℝ) := {p | p.2 = 2}
  let line_x1_sqrt3y : Set (ℝ × ℝ) := {p | p.1 = 1 + Real.sqrt 3 * p.2}

  -- Define the vertices of the triangle
  let origin : ℝ × ℝ := (0, 0)
  let vertex_on_x_axis : ℝ × ℝ := (1, 0)
  let vertex_on_y_axis : ℝ × ℝ := (0, 2)

  -- Calculate the area of the triangle
  let base : ℝ := vertex_on_x_axis.1 - origin.1
  let height : ℝ := vertex_on_y_axis.2 - origin.2
  let area : ℝ := (1 / 2) * base * height

  -- Prove that the area equals 1
  sorry

end triangle_area_l2223_222306


namespace parabola_directrix_quadratic_roots_as_eccentricities_l2223_222314

-- Define the parabola
def parabola (x y : ℝ) : Prop := x = 2 * y^2

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 5 * x + 2 = 0

-- Theorem for the parabola directrix
theorem parabola_directrix : ∃ (p : ℝ), ∀ (x y : ℝ), parabola x y → (x = -1/8 ↔ x = -p) := by sorry

-- Theorem for the quadratic equation roots as eccentricities
theorem quadratic_roots_as_eccentricities :
  ∃ (e₁ e₂ : ℝ), quadratic_equation e₁ ∧ quadratic_equation e₂ ∧
  (0 < e₁ ∧ e₁ < 1) ∧ (e₂ > 1) := by sorry

end parabola_directrix_quadratic_roots_as_eccentricities_l2223_222314


namespace sticker_difference_l2223_222392

/-- The number of stickers each person has -/
structure StickerCount where
  jerry : ℕ
  george : ℕ
  fred : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : StickerCount) : Prop :=
  s.jerry = 3 * s.george ∧
  s.george < s.fred ∧
  s.fred = 18 ∧
  s.jerry = 36

/-- The theorem to prove -/
theorem sticker_difference (s : StickerCount) 
  (h : problem_conditions s) : s.fred - s.george = 6 := by
  sorry

end sticker_difference_l2223_222392


namespace rectangle_garden_length_l2223_222357

/-- The perimeter of a rectangle -/
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: For a rectangular garden with perimeter 1800 m and breadth 400 m, the length is 500 m -/
theorem rectangle_garden_length (p b : ℝ) (h1 : p = 1800) (h2 : b = 400) :
  ∃ l : ℝ, perimeter l b = p ∧ l = 500 := by
  sorry

end rectangle_garden_length_l2223_222357


namespace vector_AB_l2223_222382

-- Define the type for 2D points
def Point := ℝ × ℝ

-- Define the vector between two points
def vector (p q : Point) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem vector_AB : 
  let A : Point := (-2, 3)
  let B : Point := (3, 2)
  vector A B = (5, -1) := by sorry

end vector_AB_l2223_222382
