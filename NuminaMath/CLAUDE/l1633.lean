import Mathlib

namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l1633_163310

theorem lcm_gcf_problem (n : ℕ+) :
  (Nat.lcm n 12 = 48) → (Nat.gcd n 12 = 8) → n = 32 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l1633_163310


namespace NUMINAMATH_CALUDE_roots_of_derivative_in_triangle_l1633_163312

open Complex

-- Define the polynomial f(x) = (x-a)(x-b)(x-c)
def f (x a b c : ℂ) : ℂ := (x - a) * (x - b) * (x - c)

-- Define the derivative of f
def f_derivative (x a b c : ℂ) : ℂ := 
  (x - b) * (x - c) + (x - a) * (x - c) + (x - a) * (x - b)

-- Define a triangle in the complex plane
def triangle_contains (a b c z : ℂ) : Prop :=
  ∃ (t1 t2 t3 : ℝ), t1 ≥ 0 ∧ t2 ≥ 0 ∧ t3 ≥ 0 ∧ t1 + t2 + t3 = 1 ∧
    z = t1 • a + t2 • b + t3 • c

-- Theorem statement
theorem roots_of_derivative_in_triangle (a b c : ℂ) :
  ∀ z : ℂ, f_derivative z a b c = 0 → triangle_contains a b c z :=
sorry

end NUMINAMATH_CALUDE_roots_of_derivative_in_triangle_l1633_163312


namespace NUMINAMATH_CALUDE_base5_20314_equals_1334_l1633_163395

def base5_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ (digits.length - 1 - i))) 0

theorem base5_20314_equals_1334 :
  base5_to_base10 [2, 0, 3, 1, 4] = 1334 := by
  sorry

end NUMINAMATH_CALUDE_base5_20314_equals_1334_l1633_163395


namespace NUMINAMATH_CALUDE_prime_power_form_l1633_163348

theorem prime_power_form (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) :
  ∃ k : ℕ, n = 3^k :=
by sorry

end NUMINAMATH_CALUDE_prime_power_form_l1633_163348


namespace NUMINAMATH_CALUDE_grocery_to_gym_speed_l1633_163360

-- Define the constants
def distance_home_to_grocery : ℝ := 840
def distance_grocery_to_gym : ℝ := 480
def time_difference : ℝ := 40

-- Define the variables
variable (speed_home_to_grocery : ℝ)
variable (speed_grocery_to_gym : ℝ)
variable (time_home_to_grocery : ℝ)
variable (time_grocery_to_gym : ℝ)

-- Define the theorem
theorem grocery_to_gym_speed :
  speed_grocery_to_gym = 2 * speed_home_to_grocery ∧
  time_home_to_grocery = distance_home_to_grocery / speed_home_to_grocery ∧
  time_grocery_to_gym = distance_grocery_to_gym / speed_grocery_to_gym ∧
  time_home_to_grocery = time_grocery_to_gym + time_difference ∧
  speed_home_to_grocery > 0 →
  speed_grocery_to_gym = 30 :=
by sorry

end NUMINAMATH_CALUDE_grocery_to_gym_speed_l1633_163360


namespace NUMINAMATH_CALUDE_average_percent_increase_l1633_163340

theorem average_percent_increase (initial_population final_population : ℕ) 
  (years : ℕ) (h1 : initial_population = 175000) (h2 : final_population = 262500) 
  (h3 : years = 10) :
  (((final_population - initial_population) / years) / initial_population) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_average_percent_increase_l1633_163340


namespace NUMINAMATH_CALUDE_intersecting_circles_theorem_l1633_163324

/-- Two circles intersecting at two distinct points theorem -/
theorem intersecting_circles_theorem 
  (r a b x₁ y₁ x₂ y₂ : ℝ) 
  (hr : r > 0)
  (hab : a ≠ 0 ∨ b ≠ 0)
  (hC₁_A : x₁^2 + y₁^2 = r^2)
  (hC₂_A : (x₁ + a)^2 + (y₁ + b)^2 = r^2)
  (hC₁_B : x₂^2 + y₂^2 = r^2)
  (hC₂_B : (x₂ + a)^2 + (y₂ + b)^2 = r^2)
  (hAB_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  (2*a*x₁ + 2*b*y₁ + a^2 + b^2 = 0) ∧ 
  (a*(x₁ - x₂) + b*(y₁ - y₂) = 0) ∧ 
  (x₁ + x₂ = -a ∧ y₁ + y₂ = -b) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_theorem_l1633_163324


namespace NUMINAMATH_CALUDE_sum_of_x_values_l1633_163337

theorem sum_of_x_values (N : ℝ) (h : N ≥ 0) : 
  ∃ x₁ x₂ : ℝ, |x₁ - 25| = N ∧ |x₂ - 25| = N ∧ x₁ + x₂ = 50 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l1633_163337


namespace NUMINAMATH_CALUDE_ball_drawing_game_l1633_163332

/-- The probability that the last ball is white in a ball-drawing game -/
def last_ball_white_probability (p q : ℕ) : ℚ :=
  if p % 2 = 0 then 0 else 1

/-- The ball-drawing game process -/
theorem ball_drawing_game (p q : ℕ) :
  let initial_total := p + q
  let final_total := 1
  let draw_count := initial_total - final_total
  ∀ (draw_process : ℕ → ℕ × ℕ),
    (∀ i < draw_count, 
      let (w, b) := draw_process i
      let (w', b') := draw_process (i + 1)
      ((w = w' ∧ b = b' + 1) ∨ (w = w' - 1 ∧ b = b' + 1) ∨ (w = w' - 2 ∧ b = b' + 1))) →
    (draw_process 0 = (p, q)) →
    (draw_process draw_count).fst + (draw_process draw_count).snd = final_total →
    (last_ball_white_probability p q = if (draw_process draw_count).fst = 1 then 1 else 0) :=
sorry

end NUMINAMATH_CALUDE_ball_drawing_game_l1633_163332


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l1633_163399

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_necessary_not_sufficient 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicularToPlane l α) 
  (h2 : subset m β) :
  (∀ (α β : Plane), parallel α β → perpendicular l m) ∧ 
  (∃ (l m : Line) (α β : Plane), 
    perpendicularToPlane l α ∧ 
    subset m β ∧ 
    perpendicular l m ∧ 
    ¬(parallel α β)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l1633_163399


namespace NUMINAMATH_CALUDE_num_valid_committees_l1633_163352

/-- Represents a community with speakers of different languages -/
structure Community where
  total : ℕ
  english : ℕ
  german : ℕ
  french : ℕ

/-- Defines a valid committee in the community -/
def ValidCommittee (c : Community) : Prop :=
  c.total = 20 ∧ c.english = 10 ∧ c.german = 10 ∧ c.french = 10

/-- Calculates the number of valid committees -/
noncomputable def NumValidCommittees (c : Community) : ℕ :=
  Nat.choose c.total 3 - Nat.choose (c.total - c.english) 3

/-- Theorem stating the number of valid committees -/
theorem num_valid_committees (c : Community) (h : ValidCommittee c) : 
  NumValidCommittees c = 1020 := by
  sorry


end NUMINAMATH_CALUDE_num_valid_committees_l1633_163352


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eleven_sqrt_two_over_six_l1633_163341

theorem sqrt_sum_equals_eleven_sqrt_two_over_six :
  Real.sqrt (9/2) + Real.sqrt (2/9) = 11 * Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eleven_sqrt_two_over_six_l1633_163341


namespace NUMINAMATH_CALUDE_grid_division_l1633_163390

/-- Represents a cell in the grid -/
inductive Cell
| Shaded
| Unshaded

/-- Represents the 6x6 grid -/
def Grid := Matrix (Fin 6) (Fin 6) Cell

/-- Counts the number of shaded cells in a given region of the grid -/
def count_shaded (g : Grid) (start_row end_row start_col end_col : Fin 6) : Nat :=
  sorry

/-- Checks if a given 3x3 region of the grid contains exactly 3 shaded cells -/
def is_valid_part (g : Grid) (start_row start_col : Fin 6) : Prop :=
  count_shaded g start_row (start_row + 2) start_col (start_col + 2) = 3

/-- The main theorem to be proved -/
theorem grid_division (g : Grid) 
  (h1 : count_shaded g 0 5 0 5 = 12) : 
  (is_valid_part g 0 0) ∧ 
  (is_valid_part g 0 3) ∧ 
  (is_valid_part g 3 0) ∧ 
  (is_valid_part g 3 3) :=
sorry

end NUMINAMATH_CALUDE_grid_division_l1633_163390


namespace NUMINAMATH_CALUDE_five_point_questions_count_l1633_163329

/-- Represents a test with two types of questions -/
structure Test where
  total_points : ℕ
  total_questions : ℕ
  five_point_questions : ℕ
  ten_point_questions : ℕ

/-- Checks if a test configuration is valid -/
def is_valid_test (t : Test) : Prop :=
  t.total_questions = t.five_point_questions + t.ten_point_questions ∧
  t.total_points = 5 * t.five_point_questions + 10 * t.ten_point_questions

theorem five_point_questions_count (t : Test) 
  (h1 : t.total_points = 200)
  (h2 : t.total_questions = 30)
  (h3 : is_valid_test t) :
  t.five_point_questions = 20 := by
  sorry

end NUMINAMATH_CALUDE_five_point_questions_count_l1633_163329


namespace NUMINAMATH_CALUDE_triangle_existence_l1633_163301

/-- Represents a triangle with side lengths and angles -/
structure Triangle where
  a : ℝ  -- base length
  b : ℝ  -- one side length
  c : ℝ  -- other side length
  α : ℝ  -- angle opposite to side a
  β : ℝ  -- angle opposite to side b
  γ : ℝ  -- angle opposite to side c

/-- The existence of a triangle with given properties -/
theorem triangle_existence 
  (a : ℝ) 
  (bc_sum : ℝ) 
  (Δθ : ℝ) 
  (h_a_pos : a > 0) 
  (h_bc_sum_pos : bc_sum > 0) 
  (h_Δθ_range : 0 < Δθ ∧ Δθ < π) :
  ∃ (t : Triangle), 
    t.a = a ∧ 
    t.b + t.c = bc_sum ∧ 
    |t.β - t.γ| = Δθ ∧
    t.α + t.β + t.γ = π :=
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l1633_163301


namespace NUMINAMATH_CALUDE_contest_scores_l1633_163358

theorem contest_scores (n k : ℕ) (hn : n ≥ 2) :
  (∀ (i : ℕ), i ≤ k → ∃! (f : ℕ → ℕ), (∀ x, x ≤ n → f x ≤ n) ∧ 
    (∀ x y, x ≠ y → f x ≠ f y) ∧ (Finset.sum (Finset.range n) f = Finset.sum (Finset.range n) id)) →
  (∀ x, x ≤ n → k * (Finset.sum (Finset.range n) id) = 26 * n) →
  (n = 25 ∧ k = 2) ∨ (n = 12 ∧ k = 4) ∨ (n = 3 ∧ k = 13) :=
by sorry

end NUMINAMATH_CALUDE_contest_scores_l1633_163358


namespace NUMINAMATH_CALUDE_number_of_tangent_lines_l1633_163345

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The hyperbola 4x^2-9y^2=36 -/
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 = 36

/-- A line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- A line has only one intersection point with the hyperbola -/
def has_one_intersection (l : Line) : Prop :=
  ∃! x y, passes_through l x y ∧ hyperbola x y

/-- The theorem to be proved -/
theorem number_of_tangent_lines : 
  ∃! (l₁ l₂ l₃ : Line), 
    (passes_through l₁ 3 0 ∧ has_one_intersection l₁) ∧
    (passes_through l₂ 3 0 ∧ has_one_intersection l₂) ∧
    (passes_through l₃ 3 0 ∧ has_one_intersection l₃) ∧
    (∀ l, passes_through l 3 0 ∧ has_one_intersection l → l = l₁ ∨ l = l₂ ∨ l = l₃) :=
sorry

end NUMINAMATH_CALUDE_number_of_tangent_lines_l1633_163345


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restrictions_l1633_163356

/-- The number of ways to seat n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to seat n people in a row where two specific people must sit together -/
def arrangementsWithPairTogether (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * (Nat.factorial 2)

/-- The number of ways to seat n people in a row where three specific people must sit together -/
def arrangementsWithTrioTogether (n : ℕ) : ℕ := (Nat.factorial (n - 2)) * (Nat.factorial 3)

/-- The number of ways to seat 7 people in a row where 3 specific people cannot sit next to each other -/
theorem seating_arrangements_with_restrictions : 
  totalArrangements 7 - 3 * arrangementsWithPairTogether 7 + arrangementsWithTrioTogether 7 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restrictions_l1633_163356


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1633_163386

theorem sqrt_equation_solution :
  let x : ℝ := 3721 / 256
  Real.sqrt x + Real.sqrt (x + 3) = 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1633_163386


namespace NUMINAMATH_CALUDE_min_xyz_l1633_163321

theorem min_xyz (x y z : ℝ) (h1 : x * y + 2 * z = 1) (h2 : x^2 + y^2 + z^2 = 10) : 
  ∀ (a b c : ℝ), a * b * c ≥ -28 → x * y * z ≥ -28 :=
by sorry

end NUMINAMATH_CALUDE_min_xyz_l1633_163321


namespace NUMINAMATH_CALUDE_first_chapter_pages_l1633_163391

/-- Represents a book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- Theorem stating the number of pages in the first chapter of the book -/
theorem first_chapter_pages (b : Book) 
  (h1 : b.chapter2_pages = 11) 
  (h2 : b.chapter1_pages = b.chapter2_pages + 37) : 
  b.chapter1_pages = 48 := by
  sorry

end NUMINAMATH_CALUDE_first_chapter_pages_l1633_163391


namespace NUMINAMATH_CALUDE_largest_number_in_set_l1633_163323

theorem largest_number_in_set : 
  let S : Set ℝ := {0.01, 0.2, 0.03, 0.02, 0.1}
  ∀ x ∈ S, x ≤ 0.2 ∧ 0.2 ∈ S := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l1633_163323


namespace NUMINAMATH_CALUDE_honeycomb_briquettes_delivery_l1633_163333

theorem honeycomb_briquettes_delivery (total : ℕ) : 
  (3 * total) / 8 + 50 = (5 * ((total - ((3 * total) / 8 + 50)))) / 7 →
  total - ((3 * total) / 8 + 50) = 700 := by
  sorry

end NUMINAMATH_CALUDE_honeycomb_briquettes_delivery_l1633_163333


namespace NUMINAMATH_CALUDE_cake_mix_distribution_l1633_163367

theorem cake_mix_distribution (first_tray second_tray total : ℕ) : 
  first_tray = second_tray + 20 →
  first_tray + second_tray = 500 →
  second_tray = 240 := by
sorry

end NUMINAMATH_CALUDE_cake_mix_distribution_l1633_163367


namespace NUMINAMATH_CALUDE_circle_C_equation_l1633_163389

def symmetric_point (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = p.2 + q.2 ∧ p.1 - q.1 = q.2 - p.2

def circle_equation (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem circle_C_equation (center : ℝ × ℝ) :
  symmetric_point center (1, 0) →
  circle_equation center 1 x y ↔ x^2 + (y - 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_C_equation_l1633_163389


namespace NUMINAMATH_CALUDE_bhanu_petrol_expenditure_l1633_163392

theorem bhanu_petrol_expenditure (income : ℝ) 
  (h1 : income > 0)
  (h2 : 0.14 * (income - 0.3 * income) = 98) : 
  0.3 * income = 300 := by
  sorry

end NUMINAMATH_CALUDE_bhanu_petrol_expenditure_l1633_163392


namespace NUMINAMATH_CALUDE_log_relationship_l1633_163342

theorem log_relationship (a b : ℝ) : 
  a = Real.log 125 / Real.log 4 → b = Real.log 32 / Real.log 5 → a = 3 * b / 10 := by
sorry

end NUMINAMATH_CALUDE_log_relationship_l1633_163342


namespace NUMINAMATH_CALUDE_f_properties_l1633_163362

noncomputable def f (x : ℝ) : ℝ := (2^x) / (4^x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f x ≤ (1/2 : ℝ)) ∧
  (∃ x : ℝ, f x = (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1633_163362


namespace NUMINAMATH_CALUDE_remainder_3056_div_32_l1633_163305

theorem remainder_3056_div_32 : 3056 % 32 = 16 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3056_div_32_l1633_163305


namespace NUMINAMATH_CALUDE_triangle_subdivision_l1633_163382

/-- Given a triangle ABC with n arbitrary non-collinear points inside it,
    the number of non-overlapping small triangles formed by connecting
    all points (including vertices A, B, C) is (2n + 1) -/
def num_small_triangles (n : ℕ) : ℕ := 2 * n + 1

/-- The main theorem stating that for 2008 points inside triangle ABC,
    the number of small triangles is 4017 -/
theorem triangle_subdivision :
  num_small_triangles 2008 = 4017 := by
  sorry

end NUMINAMATH_CALUDE_triangle_subdivision_l1633_163382


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1633_163359

theorem simplify_trig_expression :
  Real.sqrt (1 + 2 * Real.sin (π - 2) * Real.cos (π - 2)) = Real.sin 2 - Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1633_163359


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l1633_163376

theorem trajectory_is_ellipse (x y : ℝ) 
  (h1 : (2*y)^2 = (1+x)*(1-x)) 
  (h2 : y ≠ 0) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l1633_163376


namespace NUMINAMATH_CALUDE_unique_natural_number_with_specific_divisor_differences_l1633_163304

theorem unique_natural_number_with_specific_divisor_differences :
  ∃! n : ℕ,
    (∃ d₁ d₂ : ℕ, d₁ ∣ n ∧ d₂ ∣ n ∧ d₁ < d₂ ∧ ∀ d : ℕ, d ∣ n → d = d₁ ∨ d ≥ d₂) ∧
    (d₂ - d₁ = 4) ∧
    (∃ d₃ d₄ : ℕ, d₃ ∣ n ∧ d₄ ∣ n ∧ d₃ < d₄ ∧ ∀ d : ℕ, d ∣ n → d ≤ d₃ ∨ d = d₄) ∧
    (d₄ - d₃ = 308) ∧
    n = 385 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_with_specific_divisor_differences_l1633_163304


namespace NUMINAMATH_CALUDE_range_of_m_l1633_163354

-- Define the equation
def equation (m x : ℝ) : Prop := (m - 1) / (x + 1) = 1

-- Define the theorem
theorem range_of_m (m x : ℝ) : 
  equation m x ∧ x < 0 → m < 2 ∧ m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1633_163354


namespace NUMINAMATH_CALUDE_median_name_length_and_syllables_l1633_163388

theorem median_name_length_and_syllables :
  let total_names : ℕ := 23
  let names_4_1 : ℕ := 8  -- 8 names of length 4 and 1 syllable
  let names_5_2 : ℕ := 5  -- 5 names of length 5 and 2 syllables
  let names_3_1 : ℕ := 3  -- 3 names of length 3 and 1 syllable
  let names_6_2 : ℕ := 4  -- 4 names of length 6 and 2 syllables
  let names_7_3 : ℕ := 3  -- 3 names of length 7 and 3 syllables
  
  let median_position : ℕ := (total_names + 1) / 2
  
  let lengths : List ℕ := [3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7]
  let syllables : List ℕ := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]
  
  (median_position = 12) ∧
  (lengths.get! (median_position - 1) = 5) ∧
  (syllables.get! (median_position - 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_median_name_length_and_syllables_l1633_163388


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1633_163338

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 7 players, where each player plays every other player once,
    the total number of games played is 21. --/
theorem chess_tournament_games :
  num_games 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1633_163338


namespace NUMINAMATH_CALUDE_expression_simplification_l1633_163368

theorem expression_simplification (m n : ℝ) 
  (hm : m = (400 : ℝ) ^ (1/4))
  (hn : n = (5 : ℝ) ^ (1/2)) :
  ((2 - n) / (n - 1) + 4 * (m - 1) / (m - 2)) / 
  (n^2 * (m - 1) / (n - 1) + m^2 * (2 - n) / (m - 2)) = 
  ((5 : ℝ) ^ (1/2)) / 5 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1633_163368


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l1633_163344

theorem intersection_point_k_value (k : ℝ) : 
  (∃ y : ℝ, -3 * (-9.6) + 2 * y = k ∧ 0.25 * (-9.6) + y = 16) → k = 65.6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l1633_163344


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_five_plus_sqrt_two_l1633_163311

theorem sum_of_fractions_equals_five_plus_sqrt_two :
  let S := 1 / (5 - Real.sqrt 19) - 1 / (Real.sqrt 19 - Real.sqrt 18) + 
           1 / (Real.sqrt 18 - Real.sqrt 17) - 1 / (Real.sqrt 17 - 3) + 
           1 / (3 - Real.sqrt 2)
  S = 5 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_five_plus_sqrt_two_l1633_163311


namespace NUMINAMATH_CALUDE_floor_abs_neg_57_6_l1633_163302

theorem floor_abs_neg_57_6 : ⌊|(-57.6 : ℝ)|⌋ = 57 := by sorry

end NUMINAMATH_CALUDE_floor_abs_neg_57_6_l1633_163302


namespace NUMINAMATH_CALUDE_equation_solutions_l1633_163363

theorem equation_solutions :
  (∀ x : ℝ, (x - 5)^2 ≠ -1) ∧
  (∀ x : ℝ, |(-2 * x)| + 7 ≠ 0) ∧
  (∃ x : ℝ, Real.sqrt (2 - x) - 3 = 0) ∧
  (∃ x : ℝ, Real.sqrt (2 * x + 6) - 5 = 0) ∧
  (∃ x : ℝ, |(-2 * x)| - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1633_163363


namespace NUMINAMATH_CALUDE_potato_division_l1633_163355

theorem potato_division (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) :
  total_potatoes = 24 →
  num_people = 3 →
  total_potatoes = num_people * potatoes_per_person →
  potatoes_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_potato_division_l1633_163355


namespace NUMINAMATH_CALUDE_percentage_to_new_school_l1633_163396

theorem percentage_to_new_school (total_students : ℕ) 
  (percent_to_A : ℚ) (percent_to_B : ℚ) 
  (percent_A_to_C : ℚ) (percent_B_to_C : ℚ) :
  percent_to_A = 60 / 100 →
  percent_to_B = 40 / 100 →
  percent_A_to_C = 30 / 100 →
  percent_B_to_C = 40 / 100 →
  let students_A := (percent_to_A * total_students).floor
  let students_B := (percent_to_B * total_students).floor
  let students_A_to_C := (percent_A_to_C * students_A).floor
  let students_B_to_C := (percent_B_to_C * students_B).floor
  let total_to_C := students_A_to_C + students_B_to_C
  ((total_to_C : ℚ) / total_students * 100).floor = 34 := by
sorry

end NUMINAMATH_CALUDE_percentage_to_new_school_l1633_163396


namespace NUMINAMATH_CALUDE_five_object_circle_no_opposite_l1633_163315

/-- Represents a circular arrangement of n distinct objects -/
def CircularArrangement (n : ℕ) := Fin n → Fin n

/-- Two positions in a circular arrangement are opposite if they are halfway around the circle -/
def areOpposite (n : ℕ) (a b : Fin n) : Prop :=
  (a.val + n / 2) % n = b.val

/-- The probability of two specific objects being opposite in a circular arrangement of n objects -/
def probabilityOpposite (n : ℕ) : ℚ :=
  if n % 2 = 0 then 1 / n else 0

theorem five_object_circle_no_opposite :
  probabilityOpposite 5 = 0 := by
  sorry

#eval probabilityOpposite 5

end NUMINAMATH_CALUDE_five_object_circle_no_opposite_l1633_163315


namespace NUMINAMATH_CALUDE_find_unknown_number_l1633_163351

theorem find_unknown_number (n : ℕ) : 
  (∀ m : ℕ, m < 3555 → ¬(711 ∣ m ∧ n ∣ m)) → 
  (711 ∣ 3555 ∧ n ∣ 3555) → 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_find_unknown_number_l1633_163351


namespace NUMINAMATH_CALUDE_smallest_n_with_hex_digit_greater_than_9_l1633_163300

/-- Sum of digits in base b representation of n -/
def sumDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-five representation of n -/
def f (n : ℕ) : ℕ := sumDigits n 5

/-- g(n) is the sum of digits in base-nine representation of f(n) -/
def g (n : ℕ) : ℕ := sumDigits (f n) 9

/-- Converts a natural number to its base-sixteen representation -/
def toBase16 (n : ℕ) : List ℕ := sorry

/-- Checks if a list of digits contains only elements from 0 to 9 -/
def onlyDecimalDigits (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d ≤ 9

theorem smallest_n_with_hex_digit_greater_than_9 :
  (∀ m < 621, onlyDecimalDigits (toBase16 (g m))) ∧
  ¬onlyDecimalDigits (toBase16 (g 621)) := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_hex_digit_greater_than_9_l1633_163300


namespace NUMINAMATH_CALUDE_count_satisfying_numbers_l1633_163343

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ n + reverse_digits n = 145

theorem count_satisfying_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_condition n) ∧ s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_count_satisfying_numbers_l1633_163343


namespace NUMINAMATH_CALUDE_triangle_area_l1633_163327

/-- The area of the triangle formed by the x-axis, y-axis, and the line 3x + ay = 12 is 3/2 square units. -/
theorem triangle_area (a : ℝ) : 
  let x_intercept : ℝ := 12 / 3
  let y_intercept : ℝ := 12 / a
  let triangle_area : ℝ := (1 / 2) * x_intercept * y_intercept
  triangle_area = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1633_163327


namespace NUMINAMATH_CALUDE_tiktok_twitter_ratio_l1633_163330

/-- Represents the number of followers on different social media platforms --/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Calculates the total number of followers across all platforms --/
def total_followers (f : Followers) : ℕ :=
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube

/-- Theorem stating the relationship between TikTok and Twitter followers --/
theorem tiktok_twitter_ratio (f : Followers) (x : ℕ) : 
  f.instagram = 240 →
  f.facebook = 500 →
  f.twitter = (f.instagram + f.facebook) / 2 →
  f.tiktok = x * f.twitter →
  f.youtube = f.tiktok + 510 →
  total_followers f = 3840 →
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_tiktok_twitter_ratio_l1633_163330


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l1633_163366

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- The perimeter of the trapezoid -/
  perimeter : ℝ
  /-- The diagonal bisects the obtuse angle -/
  diagonal_bisects_obtuse_angle : Bool

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 96 -/
theorem isosceles_trapezoid_area :
  ∀ t : IsoscelesTrapezoid,
    t.shorter_base = 3 ∧
    t.perimeter = 42 ∧
    t.diagonal_bisects_obtuse_angle = true →
    area t = 96 :=
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l1633_163366


namespace NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_negative_two_l1633_163370

theorem tan_theta_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (2 * Real.cos θ) / (Real.sin (π/2 + θ) + Real.sin (π + θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_negative_two_l1633_163370


namespace NUMINAMATH_CALUDE_temperature_reaches_target_l1633_163372

/-- The temperature model as a function of time -/
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The target temperature -/
def target_temp : ℝ := 80

/-- The latest time when the temperature reaches the target -/
def latest_time : ℝ := 10

theorem temperature_reaches_target :
  (∃ t : ℝ, temperature t = target_temp) ∧
  (∀ t : ℝ, temperature t = target_temp → t ≤ latest_time) ∧
  (temperature latest_time = target_temp) := by
  sorry

end NUMINAMATH_CALUDE_temperature_reaches_target_l1633_163372


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_negative_four_l1633_163378

theorem at_least_one_not_greater_than_negative_four
  (a b c : ℝ)
  (ha : a < 0)
  (hb : b < 0)
  (hc : c < 0) :
  (a + 4 / b ≤ -4) ∨ (b + 4 / c ≤ -4) ∨ (c + 4 / a ≤ -4) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_negative_four_l1633_163378


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l1633_163377

-- Define the equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / (5 + m) = 1 ∧ m * (5 + m) < 0

-- State the theorem
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ -5 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l1633_163377


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1633_163387

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (b/a = Real.sqrt 2) →                   -- Slope of asymptote
  (a^2 + b^2 = 3) →                       -- Right focus coincides with parabola focus
  (∀ (x y : ℝ), x^2 - y^2/2 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1633_163387


namespace NUMINAMATH_CALUDE_prob_B_wins_value_l1633_163339

/-- The probability of player B winning in a chess game -/
def prob_B_wins (prob_A_wins : ℝ) (prob_draw : ℝ) : ℝ :=
  1 - prob_A_wins - prob_draw

/-- Theorem: The probability of player B winning is 0.3 -/
theorem prob_B_wins_value :
  prob_B_wins 0.3 0.4 = 0.3 := by
sorry

end NUMINAMATH_CALUDE_prob_B_wins_value_l1633_163339


namespace NUMINAMATH_CALUDE_wage_increase_l1633_163309

theorem wage_increase (original_wage new_wage : ℝ) (increase_percentage : ℝ) : 
  new_wage = 90 ∧ 
  increase_percentage = 50 ∧ 
  new_wage = original_wage * (1 + increase_percentage / 100) → 
  original_wage = 60 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_l1633_163309


namespace NUMINAMATH_CALUDE_solution_product_l1633_163374

theorem solution_product (r s : ℝ) : 
  (r - 3) * (3 * r + 11) = r^2 - 16 * r + 52 →
  (s - 3) * (3 * s + 11) = s^2 - 16 * s + 52 →
  r ≠ s →
  (r + 4) * (s + 4) = -62.5 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l1633_163374


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1633_163325

theorem quadratic_equation_solutions (x₁ x₂ : ℝ) :
  (x₁ = -1 ∧ x₂ = 3 ∧ x₁^2 - 2*x₁ - 3 = 0 ∧ x₂^2 - 2*x₂ - 3 = 0) →
  ∃ y₁ y₂ : ℝ, y₁ = 1 ∧ y₂ = -1 ∧ (2*y₁ + 1)^2 - 2*(2*y₁ + 1) - 3 = 0 ∧ (2*y₂ + 1)^2 - 2*(2*y₂ + 1) - 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1633_163325


namespace NUMINAMATH_CALUDE_oliver_final_amount_l1633_163375

def oliver_money (initial : ℕ) (spent : ℕ) (received : ℕ) : ℕ :=
  initial - spent + received

theorem oliver_final_amount :
  oliver_money 33 4 32 = 61 := by sorry

end NUMINAMATH_CALUDE_oliver_final_amount_l1633_163375


namespace NUMINAMATH_CALUDE_A_B_mutually_exclusive_A_C_mutually_exclusive_C_D_complementary_l1633_163303

-- Define the sample space for a six-sided die
def Ω : Type := Fin 6

-- Define the probability measure
variable (P : Ω → ℝ)

-- Assume the die is fair
axiom fair_die : ∀ x : Ω, P x = 1 / 6

-- Define events
def A (x : Ω) : Prop := x.val + 1 = 4
def B (x : Ω) : Prop := x.val % 2 = 0
def C (x : Ω) : Prop := x.val + 1 < 4
def D (x : Ω) : Prop := x.val + 1 > 3

-- Theorem statements
theorem A_B_mutually_exclusive : ∀ x : Ω, ¬(A x ∧ B x) := by sorry

theorem A_C_mutually_exclusive : ∀ x : Ω, ¬(A x ∧ C x) := by sorry

theorem C_D_complementary : ∀ x : Ω, C x ↔ ¬(D x) := by sorry

end NUMINAMATH_CALUDE_A_B_mutually_exclusive_A_C_mutually_exclusive_C_D_complementary_l1633_163303


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1633_163353

-- Define sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2*x > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1633_163353


namespace NUMINAMATH_CALUDE_shells_in_morning_l1633_163322

theorem shells_in_morning (afternoon_shells : ℕ) (total_shells : ℕ) 
  (h1 : afternoon_shells = 324)
  (h2 : total_shells = 616) :
  total_shells - afternoon_shells = 292 := by
  sorry

end NUMINAMATH_CALUDE_shells_in_morning_l1633_163322


namespace NUMINAMATH_CALUDE_ratio_equality_l1633_163336

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (norm_abc : a^2 + b^2 + c^2 = 25)
  (norm_xyz : x^2 + y^2 + z^2 = 36)
  (dot_product : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1633_163336


namespace NUMINAMATH_CALUDE_jellybean_theorem_l1633_163347

def jellybean_problem (initial : ℕ) (samantha_took : ℕ) (shelby_ate : ℕ) : ℕ :=
  let remaining_after_samantha := initial - samantha_took
  let remaining_after_shelby := remaining_after_samantha - shelby_ate
  let total_removed := samantha_took + shelby_ate
  let shannon_added := total_removed / 2
  remaining_after_shelby + shannon_added

theorem jellybean_theorem :
  jellybean_problem 90 24 12 = 72 := by
  sorry

#eval jellybean_problem 90 24 12

end NUMINAMATH_CALUDE_jellybean_theorem_l1633_163347


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_3_l1633_163318

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem subset_implies_a_geq_3 (a : ℝ) (h : A ⊆ B a) : a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_3_l1633_163318


namespace NUMINAMATH_CALUDE_parabola_directrix_l1633_163398

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form x^2 = 2py -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem: Given a parabola x^2 = 2py (p > 0) intersected by a line with slope 1 at points A and B,
    if the x-coordinate of the midpoint of AB is 2, then the equation of the directrix is y = -1 -/
theorem parabola_directrix (par : Parabola) (A B : Point) :
  (A.x^2 = 2 * par.p * A.y) →
  (B.x^2 = 2 * par.p * B.y) →
  (B.y - A.y = B.x - A.x) →
  ((A.x + B.x) / 2 = 2) →
  (∀ (x y : ℝ), y = -1 ↔ y = -par.p / 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1633_163398


namespace NUMINAMATH_CALUDE_circus_ticket_problem_l1633_163326

/-- Circus ticket problem -/
theorem circus_ticket_problem (num_kids : ℕ) (kid_ticket_price : ℚ) (total_cost : ℚ) :
  num_kids = 6 →
  kid_ticket_price = 5 →
  total_cost = 50 →
  ∃ (num_adults : ℕ),
    num_adults = 2 ∧
    total_cost = num_kids * kid_ticket_price + num_adults * (2 * kid_ticket_price) :=
by sorry

end NUMINAMATH_CALUDE_circus_ticket_problem_l1633_163326


namespace NUMINAMATH_CALUDE_july_production_l1633_163306

/-- Calculates the mask production after a given number of months, 
    starting from an initial production and doubling each month. -/
def maskProduction (initialProduction : ℕ) (months : ℕ) : ℕ :=
  initialProduction * 2^months

/-- Theorem stating that the mask production in July (4 months after March) 
    is 48000, given an initial production of 3000 in March. -/
theorem july_production : maskProduction 3000 4 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_july_production_l1633_163306


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1633_163335

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2015 + a 2017 = π →
  ∀ x : ℝ, a 2016 * (a 2014 + a 2018) ≥ π^2 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1633_163335


namespace NUMINAMATH_CALUDE_forty_second_card_is_eight_of_spades_l1633_163313

-- Define the card suits
inductive Suit
| Hearts
| Spades

-- Define the card ranks
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

-- Define a card as a pair of rank and suit
structure Card where
  rank : Rank
  suit : Suit

-- Define the cycle of cards
def cardCycle : List Card := sorry

-- Define a function to get the nth card in the cycle
def nthCard (n : Nat) : Card := sorry

-- Theorem to prove
theorem forty_second_card_is_eight_of_spades :
  nthCard 42 = Card.mk Rank.Eight Suit.Spades := by sorry

end NUMINAMATH_CALUDE_forty_second_card_is_eight_of_spades_l1633_163313


namespace NUMINAMATH_CALUDE_perfect_square_4p_minus_3_l1633_163308

theorem perfect_square_4p_minus_3 (n p : ℕ) (hn : n > 1) (hp : p > 1) (p_prime : Nat.Prime p)
  (n_divides_p_minus_1 : n ∣ (p - 1)) (p_divides_n_cube_minus_1 : p ∣ (n^3 - 1)) :
  ∃ k : ℤ, (4 : ℤ) * p - 3 = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_4p_minus_3_l1633_163308


namespace NUMINAMATH_CALUDE_tiffany_bag_collection_l1633_163373

/-- Represents the number of bags of cans Tiffany collected over three days -/
structure BagCollection where
  monday : Nat
  nextDay : Nat
  dayAfter : Nat
  total : Nat

/-- Theorem stating that given the conditions from the problem, 
    the number of bags collected on the next day must be 3 -/
theorem tiffany_bag_collection (bc : BagCollection) 
  (h1 : bc.monday = 10)
  (h2 : bc.dayAfter = 7)
  (h3 : bc.total = 20)
  (h4 : bc.monday + bc.nextDay + bc.dayAfter = bc.total) :
  bc.nextDay = 3 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bag_collection_l1633_163373


namespace NUMINAMATH_CALUDE_percentage_calculation_l1633_163379

theorem percentage_calculation (n : ℝ) (h : n = 4800) : n * 0.5 * 0.3 * 0.15 = 108 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1633_163379


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1633_163357

theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a * b = 48) 
  (h2 : b * c = 20) 
  (h3 : c * a = 15) : 
  a * b * c = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1633_163357


namespace NUMINAMATH_CALUDE_operation_result_l1633_163381

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def op : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.four
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.four
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.two

theorem operation_result : 
  op (op Element.one Element.two) (op Element.four Element.three) = Element.four := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l1633_163381


namespace NUMINAMATH_CALUDE_gnome_count_after_removal_l1633_163307

/-- The number of gnomes in each forest and the total remaining after removal --/
theorem gnome_count_after_removal :
  let westerville : ℕ := 20
  let ravenswood : ℕ := 4 * westerville
  let greenwood : ℕ := ravenswood + ravenswood / 4
  let remaining_westerville : ℕ := westerville - westerville * 3 / 10
  let remaining_ravenswood : ℕ := ravenswood - ravenswood * 2 / 5
  let remaining_greenwood : ℕ := greenwood - greenwood / 2
  remaining_westerville + remaining_ravenswood + remaining_greenwood = 112 := by
sorry

end NUMINAMATH_CALUDE_gnome_count_after_removal_l1633_163307


namespace NUMINAMATH_CALUDE_binomial_cube_special_case_l1633_163349

theorem binomial_cube_special_case : 8^3 + 3*(8^2) + 3*8 + 1 = 729 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_special_case_l1633_163349


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1633_163364

theorem sum_of_coefficients (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 26) / (x - 3)) →
  A + B = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1633_163364


namespace NUMINAMATH_CALUDE_route_b_faster_l1633_163369

/-- Represents a route with multiple segments, each with its own distance and speed. -/
structure Route where
  segments : List (Float × Float)
  total_distance : Float

/-- Calculates the total time taken to travel a route -/
def travel_time (r : Route) : Float :=
  r.segments.foldl (fun acc (d, s) => acc + d / s) 0

/-- Route A details -/
def route_a : Route :=
  { segments := [(8, 40)], total_distance := 8 }

/-- Route B details -/
def route_b : Route :=
  { segments := [(5.5, 45), (1, 25), (0.5, 15)], total_distance := 7 }

/-- The time difference between Route A and Route B in minutes -/
def time_difference : Float :=
  travel_time route_a - travel_time route_b

theorem route_b_faster : 
  0.26 < time_difference ∧ time_difference < 0.28 :=
sorry

end NUMINAMATH_CALUDE_route_b_faster_l1633_163369


namespace NUMINAMATH_CALUDE_two_planes_division_l1633_163393

/-- Represents the possible configurations of two planes in 3D space -/
inductive PlaneConfiguration
  | Parallel
  | Intersecting

/-- Represents the number of parts that two planes divide the space into -/
def spaceDivisions (config : PlaneConfiguration) : Nat :=
  match config with
  | PlaneConfiguration.Parallel => 3
  | PlaneConfiguration.Intersecting => 4

/-- Theorem stating that two planes divide space into either 3 or 4 parts -/
theorem two_planes_division :
  ∀ (config : PlaneConfiguration), 
    (spaceDivisions config = 3 ∨ spaceDivisions config = 4) :=
by sorry

end NUMINAMATH_CALUDE_two_planes_division_l1633_163393


namespace NUMINAMATH_CALUDE_min_cuts_for_eleven_sided_polygons_l1633_163350

/-- Represents a straight-line cut on a piece of paper -/
structure Cut where
  -- Add necessary fields

/-- Represents a polygon on the table -/
structure Polygon where
  sides : ℕ

/-- Represents the state of the paper after a series of cuts -/
structure PaperState where
  polygons : List Polygon

/-- Function to apply a cut to a paper state -/
def applyCut (state : PaperState) (cut : Cut) : PaperState :=
  sorry

/-- Function to count the number of eleven-sided polygons in a paper state -/
def countElevenSidedPolygons (state : PaperState) : ℕ :=
  sorry

/-- Theorem stating the minimum number of cuts required -/
theorem min_cuts_for_eleven_sided_polygons :
  ∀ (initial : PaperState),
    (∃ (cuts : List Cut),
      cuts.length = 2015 ∧
      countElevenSidedPolygons (cuts.foldl applyCut initial) ≥ 252) ∧
    (∀ (cuts : List Cut),
      cuts.length < 2015 →
      countElevenSidedPolygons (cuts.foldl applyCut initial) < 252) :=
by
  sorry

end NUMINAMATH_CALUDE_min_cuts_for_eleven_sided_polygons_l1633_163350


namespace NUMINAMATH_CALUDE_zack_traveled_to_18_countries_l1633_163328

-- Define the number of countries each person traveled to
def george_countries : ℕ := 6
def joseph_countries : ℕ := george_countries / 2
def patrick_countries : ℕ := joseph_countries * 3
def zack_countries : ℕ := patrick_countries * 2

-- Theorem statement
theorem zack_traveled_to_18_countries :
  zack_countries = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_zack_traveled_to_18_countries_l1633_163328


namespace NUMINAMATH_CALUDE_equal_digit_probability_l1633_163314

def num_dice : ℕ := 5
def sides_per_die : ℕ := 20
def one_digit_sides : ℕ := 9
def two_digit_sides : ℕ := 11

theorem equal_digit_probability : 
  (num_dice.choose (num_dice / 2)) * 
  ((two_digit_sides : ℚ) / sides_per_die) ^ (num_dice / 2) * 
  ((one_digit_sides : ℚ) / sides_per_die) ^ (num_dice - num_dice / 2) = 
  1062889 / 128000000 := by sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l1633_163314


namespace NUMINAMATH_CALUDE_ron_height_is_13_l1633_163346

/-- The height of Ron in feet -/
def ron_height : ℝ := 13

/-- The height of Dean in feet -/
def dean_height : ℝ := ron_height + 4

/-- The depth of the water in feet -/
def water_depth : ℝ := 255

theorem ron_height_is_13 :
  (water_depth = 15 * dean_height) →
  (dean_height = ron_height + 4) →
  (water_depth = 255) →
  ron_height = 13 := by
sorry

end NUMINAMATH_CALUDE_ron_height_is_13_l1633_163346


namespace NUMINAMATH_CALUDE_red_star_company_profit_optimization_l1633_163385

/-- Red Star Company's profit optimization problem -/
theorem red_star_company_profit_optimization :
  -- Define the cost per item
  let cost : ℝ := 40
  -- Define the initial sales volume (in thousand items)
  let initial_sales : ℝ := 5
  -- Define the price-sales relationship function
  let sales (x : ℝ) : ℝ :=
    if x ≤ 50 then initial_sales else 10 - 0.1 * x
  -- Define the profit function without donation
  let profit (x : ℝ) : ℝ := (x - cost) * sales x
  -- Define the profit function with donation
  let profit_with_donation (x a : ℝ) : ℝ := (x - cost - a) * sales x
  -- State the conditions and the theorem
  ∀ x a : ℝ,
    cost ≤ x ∧ x ≤ 100 →
    -- Maximum profit occurs at x = 70
    profit 70 = 90 ∧
    -- Maximum profit is 90 million yuan
    (∀ y, cost ≤ y ∧ y ≤ 100 → profit y ≤ 90) ∧
    -- With donation a = 4, maximum profit is 78 million yuan
    (x ≤ 70 → profit_with_donation x 4 ≤ 78) ∧
    profit_with_donation 70 4 = 78 := by
  sorry

end NUMINAMATH_CALUDE_red_star_company_profit_optimization_l1633_163385


namespace NUMINAMATH_CALUDE_probability_shortest_diagonal_decagon_l1633_163317

/-- The number of sides in a regular decagon -/
def n : ℕ := 10

/-- The total number of diagonals in a regular decagon -/
def total_diagonals : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular decagon -/
def shortest_diagonals : ℕ := n

/-- The probability of selecting one of the shortest diagonals -/
def probability : ℚ := shortest_diagonals / total_diagonals

theorem probability_shortest_diagonal_decagon :
  probability = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_shortest_diagonal_decagon_l1633_163317


namespace NUMINAMATH_CALUDE_alternating_sum_squares_l1633_163384

/-- The sum of squares with alternating signs in pairs from 1 to 120 -/
def M : ℕ → ℕ
| 0 => 0
| (n + 1) => if n % 4 < 2
              then M n + (120 - n + 1)^2
              else M n - (120 - n + 1)^2

theorem alternating_sum_squares : M 120 = 14520 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_squares_l1633_163384


namespace NUMINAMATH_CALUDE_lyle_friends_sandwiches_and_juice_l1633_163316

theorem lyle_friends_sandwiches_and_juice 
  (sandwich_cost : ℚ)
  (juice_cost : ℚ)
  (lyle_money : ℚ)
  (h1 : sandwich_cost = 30/100)
  (h2 : juice_cost = 20/100)
  (h3 : lyle_money = 250/100) :
  ⌊(lyle_money / (sandwich_cost + juice_cost))⌋ - 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_lyle_friends_sandwiches_and_juice_l1633_163316


namespace NUMINAMATH_CALUDE_divisor_problem_l1633_163397

theorem divisor_problem (n : ℤ) : ∃ (d : ℤ), d = 22 ∧ ∃ (k : ℤ), n = k * d + 12 ∧ ∃ (m : ℤ), 2 * n = 11 * m + 2 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l1633_163397


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1633_163320

/-- Given a polynomial Q with Q(10) = 5 and Q(50) = 15, 
    the remainder when Q is divided by (x - 10)(x - 50) is (1/4)x + 2.5 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 10 = 5) (h2 : Q 50 = 15) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 10) * (x - 50) * R x + 1/4 * x + 5/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1633_163320


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1633_163383

/-- For a geometric sequence with common ratio 1/2, prove that the ratio of the sum of first 4 terms to the 4th term is 15 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) = (1 / 2 : ℝ) * a n) →  -- Common ratio is 1/2
  (∀ n, S n = (a 1) * (1 - (1 / 2 : ℝ)^n) / (1 - (1 / 2 : ℝ))) → -- Sum formula
  S 4 / a 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1633_163383


namespace NUMINAMATH_CALUDE_inequality_proof_l1633_163394

theorem inequality_proof (a b c : ℝ) 
  (sum_cond : a + b + c = 3)
  (nonzero_cond : (6*a + b^2 + c^2) * (6*b + c^2 + a^2) * (6*c + a^2 + b^2) ≠ 0) :
  a / (6*a + b^2 + c^2) + b / (6*b + c^2 + a^2) + c / (6*c + a^2 + b^2) ≤ 3/8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1633_163394


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1633_163365

theorem decimal_to_fraction : 
  ∀ (n : ℕ), (3 : ℚ) / 10 + (24 : ℚ) / (99 * 10^n) = 19 / 33 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1633_163365


namespace NUMINAMATH_CALUDE_square_expansion_area_increase_l1633_163371

theorem square_expansion_area_increase (a : ℝ) : 
  (a + 2)^2 - a^2 = 4*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_square_expansion_area_increase_l1633_163371


namespace NUMINAMATH_CALUDE_train_distance_difference_l1633_163380

/-- Proves that the difference in distance traveled by two trains is 70 km -/
theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 20) 
  (h2 : v2 = 25) 
  (h3 : total_distance = 630) : ∃ (t : ℝ), v2 * t - v1 * t = 70 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_difference_l1633_163380


namespace NUMINAMATH_CALUDE_boat_journey_l1633_163361

-- Define the given constants
def total_time : ℝ := 19
def stream_velocity : ℝ := 4
def boat_speed : ℝ := 14

-- Define the distance between A and B
def distance_AB : ℝ := 122.14

-- Theorem statement
theorem boat_journey :
  let downstream_speed := boat_speed + stream_velocity
  let upstream_speed := boat_speed - stream_velocity
  total_time = distance_AB / downstream_speed + (distance_AB / 2) / upstream_speed :=
by
  sorry

#check boat_journey

end NUMINAMATH_CALUDE_boat_journey_l1633_163361


namespace NUMINAMATH_CALUDE_investment_percentage_l1633_163334

/-- Proves that given a sum of 4000 Rs invested at 18% p.a. for two years yields 480 Rs more in interest
    than if it were invested at x% p.a. for the same period, x must equal 12%. -/
theorem investment_percentage (x : ℝ) : 
  (4000 * 18 * 2 / 100 - 4000 * x * 2 / 100 = 480) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_l1633_163334


namespace NUMINAMATH_CALUDE_diff_sums_1500_l1633_163319

/-- Sum of the first n odd natural numbers -/
def sumOddNaturals (n : ℕ) : ℕ := n * n

/-- Sum of the first n even natural numbers -/
def sumEvenNaturals (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even natural numbers (each reduced by 3) 
    and the sum of the first n odd natural numbers -/
def diffSums (n : ℕ) : ℤ :=
  (sumEvenNaturals n - 3 * n : ℤ) - sumOddNaturals n

theorem diff_sums_1500 : diffSums 1500 = -2250 := by
  sorry

#eval diffSums 1500

end NUMINAMATH_CALUDE_diff_sums_1500_l1633_163319


namespace NUMINAMATH_CALUDE_temperature_problem_l1633_163331

/-- Given the average temperatures for two sets of three consecutive days and the temperature of the last day, prove the temperature of the first day. -/
theorem temperature_problem (T W Th F : ℝ) 
  (h1 : (T + W + Th) / 3 = 42)
  (h2 : (W + Th + F) / 3 = 44)
  (h3 : F = 43) :
  T = 37 := by
  sorry

end NUMINAMATH_CALUDE_temperature_problem_l1633_163331
