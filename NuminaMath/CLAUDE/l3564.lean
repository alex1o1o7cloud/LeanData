import Mathlib

namespace value_of_T_l3564_356498

theorem value_of_T : ∃ T : ℝ, (1/3 : ℝ) * (1/6 : ℝ) * T = (1/4 : ℝ) * (1/5 : ℝ) * 120 ∧ T = 108 := by
  sorry

end value_of_T_l3564_356498


namespace example_quadratic_function_l3564_356409

/-- A function f: ℝ → ℝ is quadratic if there exist constants a, b, c where a ≠ 0 such that
    f(x) = ax^2 + bx + c for all x ∈ ℝ -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 3x^2 + x - 1 is quadratic -/
theorem example_quadratic_function :
  IsQuadratic (fun x => 3 * x^2 + x - 1) := by
  sorry

end example_quadratic_function_l3564_356409


namespace estate_value_l3564_356434

def estate_problem (E : ℝ) : Prop :=
  let younger_son := E / 5
  let elder_son := 2 * younger_son
  let husband := 3 * younger_son
  let charity := 4000
  (younger_son + elder_son = 3 * E / 5) ∧
  (elder_son = 2 * younger_son) ∧
  (husband = 3 * younger_son) ∧
  (E = younger_son + elder_son + husband + charity)

theorem estate_value : ∃ E : ℝ, estate_problem E ∧ E = 20000 := by
  sorry

end estate_value_l3564_356434


namespace swallow_weight_ratio_l3564_356495

/-- The weight an American swallow can carry -/
def american_weight : ℝ := 5

/-- The total number of swallows in the flock -/
def total_swallows : ℕ := 90

/-- The ratio of American swallows to European swallows -/
def american_to_european_ratio : ℕ := 2

/-- The maximum combined weight the flock can carry -/
def total_weight : ℝ := 600

/-- The weight a European swallow can carry -/
def european_weight : ℝ := 10

theorem swallow_weight_ratio : 
  european_weight / american_weight = 2 := by sorry

end swallow_weight_ratio_l3564_356495


namespace ceiling_sum_sqrt_l3564_356413

theorem ceiling_sum_sqrt : ⌈Real.sqrt 19⌉ + ⌈Real.sqrt 57⌉ + ⌈Real.sqrt 119⌉ = 24 := by
  sorry

end ceiling_sum_sqrt_l3564_356413


namespace factorization_of_2x_squared_minus_50_l3564_356464

theorem factorization_of_2x_squared_minus_50 (x : ℝ) : 2 * x^2 - 50 = 2 * (x + 5) * (x - 5) := by
  sorry

end factorization_of_2x_squared_minus_50_l3564_356464


namespace expression_simplification_l3564_356461

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) := by
sorry

end expression_simplification_l3564_356461


namespace sequence_formula_l3564_356422

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 2^(n-1)) :
  ∀ n : ℕ, n > 0 → a n = 2^n - 1 := by
sorry

end sequence_formula_l3564_356422


namespace inequality_proof_l3564_356418

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : b^2 / a ≥ 2*b - a := by
  sorry

end inequality_proof_l3564_356418


namespace cone_slant_height_l3564_356448

/-- The slant height of a cone given its base circumference and lateral surface sector angle -/
theorem cone_slant_height (base_circumference : ℝ) (sector_angle : ℝ) : 
  base_circumference = 2 * Real.pi → sector_angle = 120 → 3 = 
    (base_circumference * 180) / (sector_angle * Real.pi) := by
  sorry

end cone_slant_height_l3564_356448


namespace inequality_proof_l3564_356467

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end inequality_proof_l3564_356467


namespace equation_solution_l3564_356445

theorem equation_solution : 
  ∃ x : ℝ, |Real.sqrt (x^2 + 8*x + 20) + Real.sqrt (x^2 - 2*x + 2)| = Real.sqrt 26 ∧ x = 6 := by
  sorry

end equation_solution_l3564_356445


namespace purple_walls_count_l3564_356490

theorem purple_walls_count (total_rooms : ℕ) (walls_per_room : ℕ) (green_ratio : ℚ) : 
  total_rooms = 10 → 
  walls_per_room = 8 → 
  green_ratio = 3/5 → 
  (total_rooms - total_rooms * green_ratio) * walls_per_room = 32 := by
sorry

end purple_walls_count_l3564_356490


namespace perpendicular_tangents_intersection_l3564_356458

theorem perpendicular_tangents_intersection (a b : ℝ) :
  (2*a) * (2*b) = -1 →
  let A : ℝ × ℝ := (a, a^2)
  let B : ℝ × ℝ := (b, b^2)
  let P : ℝ × ℝ := ((a + b)/2, a*b)
  (P.2 = -1/4) := by sorry

end perpendicular_tangents_intersection_l3564_356458


namespace range_of_A_l3564_356429

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_A : ∀ a : ℝ, a ∈ A ↔ a ∈ Set.Icc (-1) 3 := by sorry

end range_of_A_l3564_356429


namespace largest_five_digit_congruent_to_31_mod_26_l3564_356481

theorem largest_five_digit_congruent_to_31_mod_26 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n ≡ 31 [MOD 26] → n ≤ 99975 :=
by sorry

end largest_five_digit_congruent_to_31_mod_26_l3564_356481


namespace smallest_modulus_of_z_l3564_356455

theorem smallest_modulus_of_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z + 3*I) = 15) :
  Complex.abs z ≥ 8/5 := by
  sorry

end smallest_modulus_of_z_l3564_356455


namespace first_berry_count_l3564_356400

/-- A sequence of berry counts where the difference between consecutive counts increases by 2 -/
def BerrySequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = (a (n + 1) - a n) + 2

theorem first_berry_count
  (a : ℕ → ℕ)
  (h_seq : BerrySequence a)
  (h_2 : a 2 = 4)
  (h_3 : a 3 = 7)
  (h_4 : a 4 = 12)
  (h_5 : a 5 = 19) :
  a 1 = 3 := by
  sorry

end first_berry_count_l3564_356400


namespace smallest_cookie_boxes_l3564_356474

theorem smallest_cookie_boxes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (15 * m - 3) % 11 = 0 → n ≤ m) ∧ 
  (15 * n - 3) % 11 = 0 := by
  sorry

end smallest_cookie_boxes_l3564_356474


namespace zacks_countries_l3564_356450

theorem zacks_countries (alex george joseph patrick zack : ℕ) : 
  alex = 24 →
  george = alex / 4 →
  joseph = george / 2 →
  patrick = joseph * 3 →
  zack = patrick * 2 →
  zack = 18 :=
by sorry

end zacks_countries_l3564_356450


namespace max_value_sum_fractions_l3564_356497

theorem max_value_sum_fractions (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) :
  (a * b) / (a + b) + (a * c) / (a + c) + (a * d) / (a + d) +
  (b * c) / (b + c) + (b * d) / (b + d) + (c * d) / (c + d) ≤ 1 / 2 :=
by sorry

end max_value_sum_fractions_l3564_356497


namespace cricket_team_age_difference_l3564_356479

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  size : ℕ
  captainAge : ℕ
  wicketKeeperAge : ℕ
  averageAge : ℚ
  remainingAverageAge : ℚ

/-- The age difference between the wicket keeper and the captain -/
def ageDifference (team : CricketTeam) : ℕ :=
  team.wicketKeeperAge - team.captainAge

/-- Theorem stating the properties of the cricket team and the age difference -/
theorem cricket_team_age_difference (team : CricketTeam) 
  (h1 : team.size = 11)
  (h2 : team.captainAge = 26)
  (h3 : team.wicketKeeperAge > team.captainAge)
  (h4 : team.averageAge = 24)
  (h5 : team.remainingAverageAge = team.averageAge - 1)
  : ageDifference team = 5 := by
  sorry


end cricket_team_age_difference_l3564_356479


namespace guanaco_numbers_l3564_356436

def is_guanaco (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    a ≠ 0 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = a * 1000 + b * 100 + c * 10 + d ∧
    (a * 10 + b) * (c * 10 + d) ∣ n

theorem guanaco_numbers :
  ∀ n : ℕ, is_guanaco n ↔ (n = 1352 ∨ n = 1734) :=
by sorry

end guanaco_numbers_l3564_356436


namespace line_perpendicular_transitive_parallel_lines_from_parallel_planes_not_always_parallel_transitive_not_always_parallel_from_intersections_l3564_356415

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Theorem 1
theorem line_perpendicular_transitive 
  (l m : Line) (α : Plane) :
  parallel m l → perpendicular m α → perpendicular l α :=
sorry

-- Theorem 2
theorem parallel_lines_from_parallel_planes 
  (l m : Line) (α β γ : Plane) :
  intersect α γ m → intersect β γ l → plane_parallel α β → parallel m l :=
sorry

-- Theorem 3
theorem not_always_parallel_transitive 
  (l m : Line) (α : Plane) :
  ¬(∀ l m α, parallel m l → parallel m α → parallel l α) :=
sorry

-- Theorem 4
theorem not_always_parallel_from_intersections 
  (l m n : Line) (α β γ : Plane) :
  ¬(∀ l m n α β γ, 
    intersect α β l → intersect β γ m → intersect γ α n → 
    parallel l m ∧ parallel m n ∧ parallel l n) :=
sorry

end line_perpendicular_transitive_parallel_lines_from_parallel_planes_not_always_parallel_transitive_not_always_parallel_from_intersections_l3564_356415


namespace sock_order_ratio_l3564_356494

theorem sock_order_ratio (red_socks green_socks : ℕ) (price_green : ℝ) :
  red_socks = 5 →
  (red_socks * (3 * price_green) + green_socks * price_green) * 1.8 =
    green_socks * (3 * price_green) + red_socks * price_green →
  green_socks = 18 :=
by sorry

end sock_order_ratio_l3564_356494


namespace special_polygon_is_heptagon_l3564_356499

/-- A polygon where all diagonals passing through one vertex divide it into 5 triangles -/
structure SpecialPolygon where
  /-- The number of triangles formed by diagonals passing through one vertex -/
  num_triangles : ℕ
  /-- The number of triangles is exactly 5 -/
  h_triangles : num_triangles = 5

/-- The number of sides in a SpecialPolygon -/
def SpecialPolygon.num_sides (p : SpecialPolygon) : ℕ :=
  p.num_triangles + 2

theorem special_polygon_is_heptagon (p : SpecialPolygon) : p.num_sides = 7 := by
  sorry

end special_polygon_is_heptagon_l3564_356499


namespace quadratic_root_problem_l3564_356487

theorem quadratic_root_problem (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*k*x + k - 1 = 0 ∧ x = 0) → 
  (∃ y : ℝ, y^2 + 2*k*y + k - 1 = 0 ∧ y = -2) :=
by sorry

end quadratic_root_problem_l3564_356487


namespace complex_abs_value_l3564_356482

theorem complex_abs_value (z : ℂ) : z = (1 - Complex.I)^2 / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_abs_value_l3564_356482


namespace right_triangle_condition_l3564_356414

theorem right_triangle_condition (α β γ : Real) : 
  α + β + γ = Real.pi →
  0 ≤ α ∧ α ≤ Real.pi →
  0 ≤ β ∧ β ≤ Real.pi →
  0 ≤ γ ∧ γ ≤ Real.pi →
  Real.sin γ - Real.cos α = Real.cos β →
  α = Real.pi / 2 ∨ β = Real.pi / 2 ∨ γ = Real.pi / 2 := by
sorry

end right_triangle_condition_l3564_356414


namespace line_passes_through_fixed_point_l3564_356489

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m + 2) * 0 - (m + 1) * 1 + m + 1 = 0 := by
  sorry

end line_passes_through_fixed_point_l3564_356489


namespace find_m_l3564_356411

noncomputable def f (x m c : ℝ) : ℝ :=
  if x < m then c / Real.sqrt x else c / Real.sqrt m

theorem find_m : ∃ m : ℝ, 
  (∃ c : ℝ, f 4 m c = 30 ∧ f m m c = 15) → m = 16 := by
  sorry

end find_m_l3564_356411


namespace hugo_rolls_six_given_win_l3564_356425

-- Define the number of players and sides on the die
def num_players : ℕ := 5
def num_sides : ℕ := 8

-- Define the event of Hugo winning
def hugo_wins : Set (Fin num_players → Fin num_sides) := sorry

-- Define the event of Hugo rolling a 6 on his first roll
def hugo_rolls_six : Set (Fin num_players → Fin num_sides) := sorry

-- Define the probability measure
noncomputable def P : Set (Fin num_players → Fin num_sides) → ℝ := sorry

-- Theorem statement
theorem hugo_rolls_six_given_win :
  P (hugo_rolls_six ∩ hugo_wins) / P hugo_wins = 6375 / 32768 := by sorry

end hugo_rolls_six_given_win_l3564_356425


namespace equation_solution_l3564_356412

theorem equation_solution (x : ℚ) : 
  (3 : ℚ) / 4 + 1 / x = (7 : ℚ) / 8 → x = 8 :=
by sorry

end equation_solution_l3564_356412


namespace expression_evaluation_l3564_356432

theorem expression_evaluation (x z : ℝ) (h : x = Real.sqrt z) :
  (x - 1 / x) * (Real.sqrt z + 1 / Real.sqrt z) = z - 1 / z := by
  sorry

end expression_evaluation_l3564_356432


namespace number_of_divisors_l3564_356460

theorem number_of_divisors : ∃ n : ℕ, 
  (∀ d : ℕ, d ∣ (2008^3 + (3 * 2008 * 2009) + 1)^2 ↔ d ∈ Finset.range (n + 1) ∧ d ≠ 0) ∧
  n = 91 :=
sorry

end number_of_divisors_l3564_356460


namespace problem_solution_l3564_356441

theorem problem_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^5 = w^4) 
  (h3 : z - x = 31) : 
  (w : ℤ) - y = -2351 := by
  sorry

end problem_solution_l3564_356441


namespace pyramid_base_side_length_l3564_356431

/-- Given a right pyramid with a square base, prove that the side length of the base is 6 meters 
    when the area of one lateral face is 120 square meters and the slant height is 40 meters. -/
theorem pyramid_base_side_length (lateral_face_area slant_height : ℝ) 
  (h1 : lateral_face_area = 120)
  (h2 : slant_height = 40) : 
  let base_side := lateral_face_area / (0.5 * slant_height)
  base_side = 6 := by sorry

end pyramid_base_side_length_l3564_356431


namespace tangent_slope_at_pi_over_4_l3564_356496

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sin x + Real.cos x) - 1/2

theorem tangent_slope_at_pi_over_4 :
  let df := deriv f
  df (π/4) = 1/2 := by
  sorry

end tangent_slope_at_pi_over_4_l3564_356496


namespace max_weekly_profit_l3564_356454

-- Define the price reduction x
def x : ℝ := 5

-- Define the original cost per unit
def original_cost : ℝ := 5

-- Define the original selling price per unit
def original_price : ℝ := 14

-- Define the initial weekly sales volume
def initial_volume : ℝ := 75

-- Define the proportionality constant k
def k : ℝ := 5

-- Define the increase in sales volume as a function of price reduction
def m (x : ℝ) : ℝ := k * x^2

-- Define the weekly sales profit as a function of price reduction
def y (x : ℝ) : ℝ := (original_price - x - original_cost) * (initial_volume + m x)

-- State the theorem
theorem max_weekly_profit :
  y x = 800 ∧ ∀ z, 0 ≤ z ∧ z < 9 → y z ≤ y x :=
sorry

end max_weekly_profit_l3564_356454


namespace quadratic_equation_solution_l3564_356402

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  (f 1 = 0) ∧ (f 3 = 0) ∧ (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 3) := by
  sorry

end quadratic_equation_solution_l3564_356402


namespace simplify_expression_l3564_356473

theorem simplify_expression (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  ((x + a)^4) / ((a - b)*(a - c)) + ((x + b)^4) / ((b - a)*(b - c)) + ((x + c)^4) / ((c - a)*(c - b)) = a + b + c + 4*x :=
by sorry

end simplify_expression_l3564_356473


namespace consecutive_integers_product_812_sum_57_l3564_356437

theorem consecutive_integers_product_812_sum_57 :
  ∀ n : ℕ, n > 0 ∧ n * (n + 1) = 812 → n + (n + 1) = 57 := by
  sorry

end consecutive_integers_product_812_sum_57_l3564_356437


namespace prob_different_colors_specific_l3564_356453

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  black : ℕ
  white : ℕ
  yellow : ℕ

/-- Calculates the total number of balls in a box -/
def Box.total (b : Box) : ℕ := b.red + b.black + b.white + b.yellow

/-- The probability of drawing two balls of different colors from two boxes -/
def prob_different_colors (boxA boxB : Box) : ℚ :=
  1 - (boxA.black * boxB.black + boxA.white * boxB.white : ℚ) / 
      ((boxA.total * boxB.total) : ℚ)

/-- The main theorem stating the probability of drawing different colored balls -/
theorem prob_different_colors_specific : 
  let boxA : Box := { red := 3, black := 3, white := 3, yellow := 0 }
  let boxB : Box := { red := 0, black := 2, white := 2, yellow := 2 }
  prob_different_colors boxA boxB = 7/9 := by
  sorry


end prob_different_colors_specific_l3564_356453


namespace simplify_expression_l3564_356433

theorem simplify_expression (a b : ℝ) :
  3 * a * (3 * a^3 + 2 * a^2) - 2 * a^2 * (b^2 + 1) = 9 * a^4 + 6 * a^3 - 2 * a^2 * b^2 - 2 * a^2 :=
by sorry

end simplify_expression_l3564_356433


namespace snowboard_discount_price_l3564_356459

theorem snowboard_discount_price (original_price : ℝ) (friday_discount : ℝ) (monday_discount : ℝ) :
  original_price = 100 ∧ 
  friday_discount = 0.5 ∧ 
  monday_discount = 0.3 →
  original_price * (1 - friday_discount) * (1 - monday_discount) = 35 := by
  sorry

end snowboard_discount_price_l3564_356459


namespace inequality_proof_l3564_356493

theorem inequality_proof (x : ℝ) (h : x ≥ 0) :
  1 + x^2006 ≥ (2*x)^2005 / (1+x)^2004 := by
  sorry

end inequality_proof_l3564_356493


namespace pen_sales_profit_l3564_356427

/-- Calculates the total profit and profit percent for a pen sales scenario --/
def calculate_profit_and_percent (total_pens : ℕ) (marked_price : ℚ) 
  (discount_tier1 : ℚ) (discount_tier2 : ℚ) (discount_tier3 : ℚ)
  (pens_tier1 : ℕ) (pens_tier2 : ℕ)
  (sell_discount1 : ℚ) (sell_discount2 : ℚ)
  (pens_sold1 : ℕ) (pens_sold2 : ℕ) : ℚ × ℚ :=
  sorry

theorem pen_sales_profit :
  let total_pens : ℕ := 150
  let marked_price : ℚ := 240 / 100
  let discount_tier1 : ℚ := 5 / 100
  let discount_tier2 : ℚ := 10 / 100
  let discount_tier3 : ℚ := 15 / 100
  let pens_tier1 : ℕ := 50
  let pens_tier2 : ℕ := 50
  let sell_discount1 : ℚ := 4 / 100
  let sell_discount2 : ℚ := 2 / 100
  let pens_sold1 : ℕ := 75
  let pens_sold2 : ℕ := 75
  let (profit, percent) := calculate_profit_and_percent total_pens marked_price 
    discount_tier1 discount_tier2 discount_tier3
    pens_tier1 pens_tier2
    sell_discount1 sell_discount2
    pens_sold1 pens_sold2
  profit = 2520 / 100 ∧ abs (percent - 778 / 10000) < 1 / 10000 :=
by sorry

end pen_sales_profit_l3564_356427


namespace square_sum_eq_25_l3564_356471

theorem square_sum_eq_25 (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 7) : p^2 + q^2 = 25 := by
  sorry

end square_sum_eq_25_l3564_356471


namespace pyramid_volume_l3564_356475

theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (height : ℝ) :
  base_length = 1/3 →
  base_width = 1/4 →
  height = 1 →
  (1/3) * (base_length * base_width) * height = 1/36 := by
sorry

end pyramid_volume_l3564_356475


namespace student_allowance_equation_l3564_356424

/-- The student's weekly allowance satisfies the given equation. -/
theorem student_allowance_equation (A : ℝ) : A > 0 → (3/4 : ℝ) * (1/3 : ℝ) * ((2/5 : ℝ) * A + 4) - 2 = 0 :=
by sorry

end student_allowance_equation_l3564_356424


namespace subset_implies_a_equals_one_l3564_356480

theorem subset_implies_a_equals_one :
  ∀ (a : ℝ),
  let A : Set ℝ := {0, -a}
  let B : Set ℝ := {1, a - 2, 2 * a - 2}
  A ⊆ B → a = 1 := by
sorry

end subset_implies_a_equals_one_l3564_356480


namespace sqrt_two_between_integers_l3564_356428

theorem sqrt_two_between_integers (n : ℕ+) : 
  (n : ℝ) < Real.sqrt 2 ∧ Real.sqrt 2 < (n : ℝ) + 1 → n = 1 := by
sorry

end sqrt_two_between_integers_l3564_356428


namespace typing_time_proof_l3564_356478

def typing_time (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) : ℕ :=
  document_length / (original_speed - speed_reduction)

theorem typing_time_proof (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) 
  (h1 : original_speed = 65)
  (h2 : speed_reduction = 20)
  (h3 : document_length = 810)
  (h4 : original_speed > speed_reduction) :
  typing_time original_speed speed_reduction document_length = 18 := by
  sorry

end typing_time_proof_l3564_356478


namespace baker_remaining_pastries_l3564_356446

-- Define the initial number of pastries and the number of pastries sold
def initial_pastries : ℕ := 56
def sold_pastries : ℕ := 29

-- Define the function to calculate remaining pastries
def remaining_pastries : ℕ := initial_pastries - sold_pastries

-- Theorem statement
theorem baker_remaining_pastries : remaining_pastries = 27 := by
  sorry

end baker_remaining_pastries_l3564_356446


namespace geometric_sum_first_six_terms_l3564_356483

theorem geometric_sum_first_six_terms :
  let a₀ : ℚ := 1/2
  let r : ℚ := 1/2
  let n : ℕ := 6
  let S := a₀ * (1 - r^n) / (1 - r)
  S = 63/64 := by sorry

end geometric_sum_first_six_terms_l3564_356483


namespace crayon_box_total_l3564_356419

/-- Represents the number of crayons of each color in a crayon box. -/
structure CrayonBox where
  red : ℕ
  blue : ℕ
  green : ℕ
  pink : ℕ

/-- The total number of crayons in the box. -/
def total_crayons (box : CrayonBox) : ℕ :=
  box.red + box.blue + box.green + box.pink

/-- Theorem stating the total number of crayons in the specific box configuration. -/
theorem crayon_box_total :
  ∃ (box : CrayonBox),
    box.red = 8 ∧
    box.blue = 6 ∧
    box.green = 2 * box.blue / 3 ∧
    box.pink = 6 ∧
    total_crayons box = 24 := by
  sorry

end crayon_box_total_l3564_356419


namespace negative_two_minus_six_l3564_356404

theorem negative_two_minus_six : -2 - 6 = -8 := by
  sorry

end negative_two_minus_six_l3564_356404


namespace same_terminal_side_l3564_356486

theorem same_terminal_side (π : ℝ) : ∃ (k : ℤ), (7 / 6 * π) - (-5 / 6 * π) = k * (2 * π) := by
  sorry

end same_terminal_side_l3564_356486


namespace expression_equals_one_l3564_356406

theorem expression_equals_one :
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end expression_equals_one_l3564_356406


namespace sector_area_special_case_l3564_356472

/-- The area of a circular sector with central angle 2π/3 radians and radius 2 is 4π/3. -/
theorem sector_area_special_case :
  let central_angle : Real := (2 * Real.pi) / 3
  let radius : Real := 2
  let sector_area : Real := (1 / 2) * radius^2 * central_angle
  sector_area = (4 * Real.pi) / 3 := by
  sorry

end sector_area_special_case_l3564_356472


namespace largest_divisor_l3564_356444

theorem largest_divisor (A B : ℕ) (h1 : 13 = 4 * A + B) (h2 : B < A) : A ≤ 3 := by
  sorry

end largest_divisor_l3564_356444


namespace part_one_part_two_l3564_356469

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y - 1
def B (x y : ℝ) : ℝ := x^2 - x * y

-- Part 1
theorem part_one : ∀ x y : ℝ, (x + 1)^2 + |y - 2| = 0 → A x y - 2 * B x y = -7 := by
  sorry

-- Part 2
theorem part_two : (∃ c : ℝ, ∀ x y : ℝ, A x y - 2 * B x y = c) → 
  ∃ x : ℝ, x^2 - 2*x - 1 = -1/25 := by
  sorry

end part_one_part_two_l3564_356469


namespace eric_sibling_product_l3564_356452

/-- Represents a family with a given number of sisters and brothers -/
structure Family where
  sisters : ℕ
  brothers : ℕ

/-- Calculates the number of sisters and brothers for a sibling in the family -/
def siblingCounts (f : Family) : ℕ × ℕ :=
  (f.sisters + 1, f.brothers)

theorem eric_sibling_product (emmas_family : Family)
    (h1 : emmas_family.sisters = 4)
    (h2 : emmas_family.brothers = 6) :
    let (S, B) := siblingCounts emmas_family
    S * B = 30 := by
  sorry

end eric_sibling_product_l3564_356452


namespace pastry_sets_problem_l3564_356442

theorem pastry_sets_problem (N : ℕ) 
  (h1 : ∃ (x y : ℕ), x + y = N ∧ 3*x + 5*y = 25)
  (h2 : ∃ (a b : ℕ), a + b = N ∧ 3*a + 5*b = 35) : 
  N = 7 := by
sorry

end pastry_sets_problem_l3564_356442


namespace banana_orange_equivalence_l3564_356491

/-- Given that 3/5 of 15 bananas are worth as much as 12 oranges,
    prove that 2/3 of 9 bananas are worth as much as 8 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3 / 5 : ℚ) * 15 * banana_value = 12 * orange_value →
  (2 / 3 : ℚ) * 9 * banana_value = 8 * orange_value :=
by sorry

end banana_orange_equivalence_l3564_356491


namespace negation_of_existence_irrational_square_l3564_356465

theorem negation_of_existence_irrational_square :
  (¬ ∃ x : ℝ, Irrational (x^2)) ↔ (∀ x : ℝ, ¬ Irrational (x^2)) := by
  sorry

end negation_of_existence_irrational_square_l3564_356465


namespace intersection_of_A_and_B_l3564_356420

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

-- Define set B
def B : Set ℝ := Set.range (λ x => 2 * x + 1)

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ici 1 := by sorry

end intersection_of_A_and_B_l3564_356420


namespace prove_length_l3564_356430

-- Define the points
variable (A O B A1 B1 : ℝ)

-- Define the conditions
axiom collinear : ∃ (t : ℝ), O = t • A + (1 - t) • B
axiom symmetric_A : A1 - O = O - A
axiom symmetric_B : B1 - O = O - B
axiom given_length : abs (A - B1) = 2

-- State the theorem
theorem prove_length : abs (A1 - B) = 2 := by sorry

end prove_length_l3564_356430


namespace difference_of_squares_l3564_356408

theorem difference_of_squares (m n : ℝ) : m^2 - 4*n^2 = (m + 2*n) * (m - 2*n) := by
  sorry

end difference_of_squares_l3564_356408


namespace original_number_of_people_l3564_356488

theorem original_number_of_people (x : ℕ) : 
  (x / 2 : ℚ) - (x / 2 : ℚ) / 3 = 12 → x = 36 := by
  sorry

end original_number_of_people_l3564_356488


namespace painting_price_increase_percentage_l3564_356439

/-- Proves that the percentage increase in the cost of each painting is 20% --/
theorem painting_price_increase_percentage :
  let original_jewelry_price : ℚ := 30
  let original_painting_price : ℚ := 100
  let jewelry_price_increase : ℚ := 10
  let jewelry_quantity : ℕ := 2
  let painting_quantity : ℕ := 5
  let total_cost : ℚ := 680
  let new_jewelry_price : ℚ := original_jewelry_price + jewelry_price_increase
  let painting_price_increase_percentage : ℚ := 20

  (jewelry_quantity : ℚ) * new_jewelry_price + 
  (painting_quantity : ℚ) * original_painting_price * (1 + painting_price_increase_percentage / 100) = 
  total_cost :=
by sorry

end painting_price_increase_percentage_l3564_356439


namespace trajectory_of_vertex_C_trajectory_of_vertex_C_proof_l3564_356477

/-- The trajectory of vertex C in triangle ABC, where A(0, 2) and B(0, -2), 
    and the perimeter is 10, forms an ellipse. -/
theorem trajectory_of_vertex_C (C : ℝ × ℝ) : Prop :=
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (0, -2)
  let perimeter : ℝ := 10
  let dist (P Q : ℝ × ℝ) := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  dist A B + dist B C + dist C A = perimeter →
  C.1 ≠ 0 →
  C.1^2 / 5 + C.2^2 / 9 = 1

/-- The proof of the theorem. -/
theorem trajectory_of_vertex_C_proof : ∀ C, trajectory_of_vertex_C C := by
  sorry

end trajectory_of_vertex_C_trajectory_of_vertex_C_proof_l3564_356477


namespace five_month_practice_time_l3564_356407

/-- Calculates the total piano practice time over a given number of months. -/
def total_practice_time (weekly_hours : ℕ) (weeks_per_month : ℕ) (months : ℕ) : ℕ :=
  weekly_hours * weeks_per_month * months

/-- Theorem stating that practicing 4 hours per week for 5 months results in 80 hours of practice. -/
theorem five_month_practice_time :
  total_practice_time 4 4 5 = 80 := by
  sorry

#eval total_practice_time 4 4 5

end five_month_practice_time_l3564_356407


namespace sqrt_21_bounds_l3564_356451

theorem sqrt_21_bounds : 4 < Real.sqrt 21 ∧ Real.sqrt 21 < 5 := by sorry

end sqrt_21_bounds_l3564_356451


namespace firefighter_remaining_money_is_2340_l3564_356485

/-- Calculates the remaining money for a firefighter after monthly expenses --/
def firefighter_remaining_money (hourly_rate : ℚ) (weekly_hours : ℚ) (food_expense : ℚ) (tax_expense : ℚ) : ℚ :=
  let weekly_earnings := hourly_rate * weekly_hours
  let monthly_earnings := weekly_earnings * 4
  let rent_expense := monthly_earnings / 3
  let total_expenses := rent_expense + food_expense + tax_expense
  monthly_earnings - total_expenses

/-- Theorem stating that the firefighter's remaining money is $2340 --/
theorem firefighter_remaining_money_is_2340 :
  firefighter_remaining_money 30 48 500 1000 = 2340 := by
  sorry

#eval firefighter_remaining_money 30 48 500 1000

end firefighter_remaining_money_is_2340_l3564_356485


namespace food_percentage_is_twenty_percent_l3564_356438

/-- Represents the percentage of total amount spent on each category and their respective tax rates -/
structure ShoppingExpenses where
  clothing_percent : Real
  other_percent : Real
  clothing_tax : Real
  other_tax : Real
  total_tax : Real

/-- Calculates the percentage spent on food given the shopping expenses -/
def food_percent (e : ShoppingExpenses) : Real :=
  1 - e.clothing_percent - e.other_percent

/-- Calculates the total tax rate based on the expenses and tax rates -/
def total_tax_rate (e : ShoppingExpenses) : Real :=
  e.clothing_percent * e.clothing_tax + e.other_percent * e.other_tax

/-- Theorem stating that given the shopping conditions, the percentage spent on food is 20% -/
theorem food_percentage_is_twenty_percent (e : ShoppingExpenses) 
  (h1 : e.clothing_percent = 0.5)
  (h2 : e.other_percent = 0.3)
  (h3 : e.clothing_tax = 0.04)
  (h4 : e.other_tax = 0.1)
  (h5 : e.total_tax = 0.05)
  (h6 : total_tax_rate e = e.total_tax) :
  food_percent e = 0.2 := by
  sorry

end food_percentage_is_twenty_percent_l3564_356438


namespace work_completion_time_l3564_356416

theorem work_completion_time (b : ℝ) (a_wage_ratio : ℝ) (a : ℝ) : 
  b = 15 →
  a_wage_ratio = 3/5 →
  (1/a) / ((1/a) + (1/b)) = a_wage_ratio →
  a = 10 := by
sorry

end work_completion_time_l3564_356416


namespace bike_purchase_weeks_l3564_356435

def bike_cost : ℕ := 600
def gift_money : ℕ := 150
def weekly_earnings : ℕ := 20

def weeks_needed : ℕ := 23

theorem bike_purchase_weeks : 
  ∀ (w : ℕ), w ≥ weeks_needed ↔ gift_money + w * weekly_earnings ≥ bike_cost :=
by sorry

end bike_purchase_weeks_l3564_356435


namespace courses_selection_theorem_l3564_356457

/-- The number of available courses -/
def n : ℕ := 4

/-- The number of courses each person selects -/
def k : ℕ := 2

/-- The number of ways to select courses with at least one different course -/
def select_courses : ℕ := 30

/-- Theorem stating that the number of ways to select courses with at least one different course is 30 -/
theorem courses_selection_theorem : select_courses = 30 := by
  sorry

end courses_selection_theorem_l3564_356457


namespace p_necessary_not_sufficient_for_q_l3564_356470

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 4*x + 3 > 0
def q (x : ℝ) : Prop := x^2 < 1

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end p_necessary_not_sufficient_for_q_l3564_356470


namespace unique_solution_condition_l3564_356401

theorem unique_solution_condition (a b c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = (b + 1) * x + c) ↔ b ≠ 3 := by
  sorry

end unique_solution_condition_l3564_356401


namespace isabella_currency_exchange_l3564_356443

/-- Represents the exchange of US dollars to Canadian dollars and subsequent spending -/
def exchange_and_spend (d : ℕ) : Prop :=
  (8 * d) / 5 - 75 = d

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The main theorem representing the problem -/
theorem isabella_currency_exchange :
  ∃ d : ℕ, exchange_and_spend d ∧ d = 125 ∧ sum_of_digits d = 8 :=
sorry

end isabella_currency_exchange_l3564_356443


namespace equation_has_real_roots_l3564_356403

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^3 * (x + 1) * (x - 3) := by
  sorry

end equation_has_real_roots_l3564_356403


namespace alba_oranges_theorem_l3564_356423

/-- Represents the orange production and sale scenario of the Morales sisters -/
structure OrangeScenario where
  trees_per_sister : ℕ
  gabriela_oranges_per_tree : ℕ
  maricela_oranges_per_tree : ℕ
  oranges_per_cup : ℕ
  price_per_cup : ℕ
  total_revenue : ℕ

/-- Calculates the number of oranges Alba's trees produce per tree -/
def alba_oranges_per_tree (scenario : OrangeScenario) : ℕ :=
  let total_cups := scenario.total_revenue / scenario.price_per_cup
  let total_oranges := total_cups * scenario.oranges_per_cup
  let gabriela_oranges := scenario.gabriela_oranges_per_tree * scenario.trees_per_sister
  let maricela_oranges := scenario.maricela_oranges_per_tree * scenario.trees_per_sister
  let alba_total_oranges := total_oranges - gabriela_oranges - maricela_oranges
  alba_total_oranges / scenario.trees_per_sister

/-- The main theorem stating that given the scenario conditions, Alba's trees produce 400 oranges per tree -/
theorem alba_oranges_theorem (scenario : OrangeScenario) 
  (h1 : scenario.trees_per_sister = 110)
  (h2 : scenario.gabriela_oranges_per_tree = 600)
  (h3 : scenario.maricela_oranges_per_tree = 500)
  (h4 : scenario.oranges_per_cup = 3)
  (h5 : scenario.price_per_cup = 4)
  (h6 : scenario.total_revenue = 220000) :
  alba_oranges_per_tree scenario = 400 := by
  sorry

end alba_oranges_theorem_l3564_356423


namespace simplify_expression_l3564_356447

theorem simplify_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end simplify_expression_l3564_356447


namespace sum_of_reciprocals_l3564_356417

theorem sum_of_reciprocals (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 8 * x * y) :
  1 / x + 1 / y = 8 := by
sorry

end sum_of_reciprocals_l3564_356417


namespace exam_average_l3564_356410

theorem exam_average (group1_count : ℕ) (group1_avg : ℚ) 
                      (group2_count : ℕ) (group2_avg : ℚ) : 
  group1_count = 15 →
  group1_avg = 70 / 100 →
  group2_count = 10 →
  group2_avg = 90 / 100 →
  (group1_count * group1_avg + group2_count * group2_avg) / (group1_count + group2_count) = 78 / 100 := by
  sorry

end exam_average_l3564_356410


namespace trajectory_of_moving_circle_l3564_356468

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
structure CircleConfiguration where
  O₁ : Circle
  O₂ : Circle
  O : Circle
  h₁ : O₁.radius ≠ O₂.radius
  h₂ : O₁.center ≠ O₂.center
  h₃ : ∀ p : ℝ × ℝ, (dist p O₁.center ≠ O₁.radius) ∨ (dist p O₂.center ≠ O₂.radius)
  h₄ : dist O.center O₁.center = O.radius + O₁.radius ∨ dist O.center O₁.center = abs (O.radius - O₁.radius)
  h₅ : dist O.center O₂.center = O.radius + O₂.radius ∨ dist O.center O₂.center = abs (O.radius - O₂.radius)

-- Define the trajectory types
inductive TrajectoryType
  | Hyperbola
  | Ellipse

-- State the theorem
theorem trajectory_of_moving_circle (config : CircleConfiguration) :
  ∃ t : TrajectoryType, t = TrajectoryType.Hyperbola ∨ t = TrajectoryType.Ellipse :=
sorry

end trajectory_of_moving_circle_l3564_356468


namespace total_spent_on_pens_l3564_356484

def brand_x_price : ℝ := 4.00
def brand_y_price : ℝ := 2.20
def total_pens : ℕ := 12
def brand_x_count : ℕ := 6

theorem total_spent_on_pens : 
  brand_x_count * brand_x_price + (total_pens - brand_x_count) * brand_y_price = 37.20 :=
by sorry

end total_spent_on_pens_l3564_356484


namespace smallest_two_digit_prime_with_even_reverse_l3564_356426

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ Even (reverse_digits n)

theorem smallest_two_digit_prime_with_even_reverse : 
  satisfies_condition 23 ∧ ∀ n : ℕ, satisfies_condition n → 23 ≤ n :=
sorry

end smallest_two_digit_prime_with_even_reverse_l3564_356426


namespace work_completion_time_l3564_356476

/-- The number of days it takes 'a' to complete the work -/
def days_a : ℕ := 27

/-- The number of days it takes 'b' to complete the work -/
def days_b : ℕ := 2 * days_a

theorem work_completion_time : days_a = 27 := by
  sorry

end work_completion_time_l3564_356476


namespace square_sum_value_l3564_356405

theorem square_sum_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y + x + y = 71) (h2 : x^2 * y + x * y^2 = 880) :
  x^2 + y^2 = 146 := by
sorry

end square_sum_value_l3564_356405


namespace sufficient_but_not_necessary_l3564_356462

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end sufficient_but_not_necessary_l3564_356462


namespace arithmetic_progression_of_primes_l3564_356463

theorem arithmetic_progression_of_primes (p₁ p₂ p₃ d : ℕ) : 
  Prime p₁ → Prime p₂ → Prime p₃ →  -- The numbers are prime
  p₁ > 3 → p₂ > 3 → p₃ > 3 →        -- The numbers are greater than 3
  p₁ < p₂ ∧ p₂ < p₃ →               -- The numbers are in ascending order
  p₂ = p₁ + d →                     -- Definition of arithmetic progression
  p₃ = p₁ + 2*d →                   -- Definition of arithmetic progression
  ∃ k : ℕ, d = 6 * k                -- The common difference is divisible by 6
  := by sorry

end arithmetic_progression_of_primes_l3564_356463


namespace sandy_initial_money_l3564_356440

/-- Given that Sandy spent $6 on a pie and has $57 left, prove that she initially had $63. -/
theorem sandy_initial_money :
  ∀ (initial_money spent_on_pie money_left : ℕ),
    spent_on_pie = 6 →
    money_left = 57 →
    initial_money = spent_on_pie + money_left →
    initial_money = 63 := by
  sorry

end sandy_initial_money_l3564_356440


namespace motorcycles_in_anytown_l3564_356492

/-- Represents the number of vehicles of each type in Anytown -/
structure VehicleCounts where
  trucks : ℕ
  sedans : ℕ
  motorcycles : ℕ

/-- The ratio of vehicles in Anytown -/
def vehicle_ratio : VehicleCounts := ⟨3, 7, 2⟩

/-- The actual number of sedans in Anytown -/
def actual_sedans : ℕ := 9100

/-- Theorem stating the number of motorcycles in Anytown -/
theorem motorcycles_in_anytown : 
  ∃ (vc : VehicleCounts), 
    vc.sedans = actual_sedans ∧ 
    vc.trucks * vehicle_ratio.sedans = vc.sedans * vehicle_ratio.trucks ∧
    vc.sedans * vehicle_ratio.motorcycles = vc.motorcycles * vehicle_ratio.sedans ∧
    vc.motorcycles = 2600 := by
  sorry

end motorcycles_in_anytown_l3564_356492


namespace min_value_of_abs_sum_l3564_356466

theorem min_value_of_abs_sum (x : ℝ) : 
  |x - 4| + |x - 6| ≥ 2 ∧ ∃ y : ℝ, |y - 4| + |y - 6| = 2 :=
by sorry

end min_value_of_abs_sum_l3564_356466


namespace midpoint_coordinate_sum_l3564_356456

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, 10) and (-4, -10) is 2. -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := 10
  let x₂ : ℝ := -4
  let y₂ : ℝ := -10
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 2 := by
sorry

end midpoint_coordinate_sum_l3564_356456


namespace probability_triangle_or_hexagon_l3564_356421

theorem probability_triangle_or_hexagon :
  let total_figures : ℕ := 10
  let triangles : ℕ := 3
  let squares : ℕ := 4
  let circles : ℕ := 2
  let hexagons : ℕ := 1
  let target_figures := triangles + hexagons
  (target_figures : ℚ) / total_figures = 2 / 5 :=
by sorry

end probability_triangle_or_hexagon_l3564_356421


namespace arithmetic_sqrt_16_l3564_356449

theorem arithmetic_sqrt_16 : Real.sqrt 16 = 4 := by
  sorry

end arithmetic_sqrt_16_l3564_356449
