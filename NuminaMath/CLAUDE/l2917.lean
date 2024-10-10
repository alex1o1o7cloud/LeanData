import Mathlib

namespace apollonian_circle_l2917_291793

/-- The Apollonian circle theorem -/
theorem apollonian_circle (r : ℝ) (h_r_pos : r > 0) : 
  (∃! P : ℝ × ℝ, (P.1 - 2)^2 + P.2^2 = r^2 ∧ 
    ((P.1 - 3)^2 + P.2^2) = 4 * (P.1^2 + P.2^2)) → r = 1 := by
  sorry

end apollonian_circle_l2917_291793


namespace inequality_solution_range_l2917_291786

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) → 
  (a ≤ -6 ∨ a ≥ 2) := by
sorry

end inequality_solution_range_l2917_291786


namespace parallel_vectors_x_value_l2917_291745

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x = k * w.x ∧ v.y = k * w.y

theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : Vector2D := ⟨x - 1, 2⟩
  let b : Vector2D := ⟨2, 1⟩
  parallel a b → x = 5 := by
  sorry

end parallel_vectors_x_value_l2917_291745


namespace book_selling_price_l2917_291777

/-- Proves that the selling price of each book is $1.50 --/
theorem book_selling_price (total_books : ℕ) (records_bought : ℕ) (record_price : ℚ) (money_left : ℚ) :
  total_books = 200 →
  records_bought = 75 →
  record_price = 3 →
  money_left = 75 →
  (total_books : ℚ) * (1.5 : ℚ) = records_bought * record_price + money_left :=
by
  sorry

#check book_selling_price

end book_selling_price_l2917_291777


namespace rectangle_length_l2917_291721

/-- Represents the properties of a rectangle --/
structure Rectangle where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_width_relation : length = width + 15
  perimeter_formula : perimeter = 2 * length + 2 * width

/-- Theorem stating that a rectangle with the given properties has a length of 45 cm --/
theorem rectangle_length (rect : Rectangle) (h : rect.perimeter = 150) : rect.length = 45 := by
  sorry

end rectangle_length_l2917_291721


namespace regular_polygon_sides_l2917_291773

theorem regular_polygon_sides (interior_angle : ℝ) : 
  interior_angle = 150 → (360 / (180 - interior_angle) : ℝ) = 12 :=
by sorry

end regular_polygon_sides_l2917_291773


namespace fusilli_to_penne_ratio_l2917_291723

/-- Given a survey of pasta preferences, prove the ratio of fusilli to penne preferences --/
theorem fusilli_to_penne_ratio :
  ∀ (total students_fusilli students_penne : ℕ),
  total = 800 →
  students_fusilli = 320 →
  students_penne = 160 →
  (students_fusilli : ℚ) / (students_penne : ℚ) = 2 := by
sorry

end fusilli_to_penne_ratio_l2917_291723


namespace intersection_of_A_and_B_l2917_291724

def A : Set ℤ := {-2, -1}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end intersection_of_A_and_B_l2917_291724


namespace max_sunny_days_thursday_l2917_291789

/-- Represents the days of the week -/
inductive Day : Type
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Represents the weather conditions -/
inductive Weather : Type
  | sunny
  | rainy
  | foggy

/-- The weather pattern for each day of the week -/
def weatherPattern (d : Day) : Weather :=
  match d with
  | Day.monday => Weather.rainy
  | Day.friday => Weather.rainy
  | Day.saturday => Weather.foggy
  | _ => Weather.sunny

/-- Calculates the number of sunny days in a 30-day period starting from a given day -/
def sunnyDaysCount (startDay : Day) : Nat :=
  sorry

/-- Theorem: Starting on Thursday maximizes the number of sunny days in a 30-day period -/
theorem max_sunny_days_thursday :
  ∀ d : Day, sunnyDaysCount Day.thursday ≥ sunnyDaysCount d :=
  sorry

end max_sunny_days_thursday_l2917_291789


namespace max_area_rhombus_l2917_291749

/-- Given a rhombus OABC in a rectangular coordinate system xOy with the following properties:
  - The diagonals intersect at point M(x₀, y₀)
  - The hyperbola y = k/x (x > 0) passes through points C and M
  - 2 ≤ x₀ ≤ 4
  Prove that the maximum area of rhombus OABC is 24√2 -/
theorem max_area_rhombus (x₀ y₀ k : ℝ) (hx₀ : 2 ≤ x₀ ∧ x₀ ≤ 4) (hk : k > 0) 
  (h_hyperbola : y₀ = k / x₀) : 
  (∃ (S : ℝ), S = 24 * Real.sqrt 2 ∧ 
    ∀ (A : ℝ), A ≤ S ∧ 
    (∃ (x₁ y₁ : ℝ), 2 ≤ x₁ ∧ x₁ ≤ 4 ∧ 
      y₁ = k / x₁ ∧ 
      A = (3 * Real.sqrt 2 / 2) * x₁^2)) := by
  sorry

end max_area_rhombus_l2917_291749


namespace min_value_of_a_l2917_291739

theorem min_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x > a → x^2 - x - 6 > 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 > 0 ∧ x ≤ a) → 
  a = 3 :=
sorry

end min_value_of_a_l2917_291739


namespace smallest_value_for_x_between_zero_and_one_l2917_291737

theorem smallest_value_for_x_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^3 < x^2 ∧ x^2 < x ∧ x < 2*x ∧ 2*x < 3*x := by
  sorry

end smallest_value_for_x_between_zero_and_one_l2917_291737


namespace football_players_l2917_291771

theorem football_players (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 460)
  (h2 : cricket = 175)
  (h3 : neither = 50)
  (h4 : both = 90) :
  total - neither - cricket + both = 325 :=
by sorry

end football_players_l2917_291771


namespace min_value_theorem_l2917_291791

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_constraint : 3 * m + n = 1) :
  1 / m + 3 / n ≥ 12 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 3 * m₀ + n₀ = 1 ∧ 1 / m₀ + 3 / n₀ = 12 := by
  sorry

end min_value_theorem_l2917_291791


namespace train_length_l2917_291794

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 12 → 
  ∃ length_m : ℝ, abs (length_m - 200.04) < 0.01 := by
  sorry

end train_length_l2917_291794


namespace business_class_seats_count_l2917_291747

/-- A small airplane with first, business, and economy class seating. -/
structure Airplane where
  first_class_seats : ℕ
  business_class_seats : ℕ
  economy_class_seats : ℕ

/-- Theorem stating the number of business class seats in the airplane. -/
theorem business_class_seats_count (a : Airplane) 
  (h1 : a.first_class_seats = 10)
  (h2 : a.economy_class_seats = 50)
  (h3 : a.economy_class_seats / 2 = a.first_class_seats + (a.business_class_seats - 8))
  (h4 : a.first_class_seats - 7 = 3) : 
  a.business_class_seats = 30 := by
  sorry


end business_class_seats_count_l2917_291747


namespace current_short_trees_count_l2917_291799

/-- The number of short trees currently in the park -/
def current_short_trees : ℕ := 41

/-- The number of short trees to be planted today -/
def trees_to_plant : ℕ := 57

/-- The total number of short trees after planting -/
def total_short_trees : ℕ := 98

/-- Theorem stating that the number of short trees currently in the park is 41 -/
theorem current_short_trees_count :
  current_short_trees + trees_to_plant = total_short_trees :=
by sorry

end current_short_trees_count_l2917_291799


namespace geometric_sequence_ratio_l2917_291767

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_decreasing : ∀ n, a (n + 1) < a n)
  (h_geom : geometric_sequence a)
  (h_prod : a 2 * a 8 = 6)
  (h_sum : a 4 + a 6 = 5) :
  a 5 / a 7 = 3/2 := by
sorry

end geometric_sequence_ratio_l2917_291767


namespace fish_remaining_l2917_291715

theorem fish_remaining (initial : Float) (given_away : Float) : 
  initial = 47.0 → given_away = 22.5 → initial - given_away = 24.5 := by
  sorry

end fish_remaining_l2917_291715


namespace min_draw_count_correct_l2917_291784

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to draw to guarantee the condition -/
def minDrawCount : Nat := 82

/-- The theorem stating the minimum number of balls to draw -/
theorem min_draw_count_correct (box : BallCounts) 
  (h1 : box.red = 30)
  (h2 : box.green = 22)
  (h3 : box.yellow = 20)
  (h4 : box.blue = 15)
  (h5 : box.white = 12)
  (h6 : box.black = 10) :
  minDrawCount = 82 ∧ 
  (∀ n : Nat, n < 82 → 
    ∃ draw : BallCounts, 
      draw.red + draw.green + draw.yellow + draw.blue + draw.white + draw.black = n ∧
      draw.red ≤ box.red ∧ draw.green ≤ box.green ∧ draw.yellow ≤ box.yellow ∧ 
      draw.blue ≤ box.blue ∧ draw.white ≤ box.white ∧ draw.black ≤ box.black ∧
      draw.white ≤ 12 ∧
      draw.red < 16 ∧ draw.green < 16 ∧ draw.yellow < 16 ∧ draw.blue < 16 ∧ draw.black < 16) ∧
  (∃ draw : BallCounts,
    draw.red + draw.green + draw.yellow + draw.blue + draw.white + draw.black = 82 ∧
    draw.red ≤ box.red ∧ draw.green ≤ box.green ∧ draw.yellow ≤ box.yellow ∧ 
    draw.blue ≤ box.blue ∧ draw.white ≤ box.white ∧ draw.black ≤ box.black ∧
    draw.white ≤ 12 ∧
    (draw.red ≥ 16 ∨ draw.green ≥ 16 ∨ draw.yellow ≥ 16 ∨ draw.blue ≥ 16 ∨ draw.black ≥ 16)) :=
by sorry

end min_draw_count_correct_l2917_291784


namespace complex_number_in_quadrant_IV_l2917_291702

/-- The complex number (1+i)/(1+2i) lies in Quadrant IV of the complex plane -/
theorem complex_number_in_quadrant_IV : 
  let z : ℂ := (1 + Complex.I) / (1 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end complex_number_in_quadrant_IV_l2917_291702


namespace cricket_team_average_age_l2917_291792

/-- Represents the average age of a cricket team --/
def average_age : ℝ := 23

/-- Represents the number of team members --/
def team_size : ℕ := 11

/-- Represents the age of the captain --/
def captain_age : ℕ := 25

/-- Represents the age of the wicket keeper --/
def wicket_keeper_age : ℕ := captain_age + 3

/-- Represents the age of the vice-captain --/
def vice_captain_age : ℕ := wicket_keeper_age - 4

/-- Theorem stating that the average age of the cricket team is 23 years --/
theorem cricket_team_average_age :
  average_age * team_size =
    captain_age + wicket_keeper_age + vice_captain_age +
    (team_size - 3) * (average_age - 1) := by
  sorry

#check cricket_team_average_age

end cricket_team_average_age_l2917_291792


namespace unique_polynomial_composition_l2917_291765

-- Define the polynomial P(x) = a x^2 + b x + c
def P (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define a general n-degree polynomial
def NPolynomial (n : ℕ) := ℝ → ℝ

theorem unique_polynomial_composition (a b c : ℝ) (ha : a ≠ 0) (n : ℕ) :
  ∃! Q : NPolynomial n, ∀ x : ℝ, Q (P a b c x) = P a b c (Q x) :=
sorry

end unique_polynomial_composition_l2917_291765


namespace tangent_perpendicular_implies_a_value_l2917_291746

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a^x + 1

theorem tangent_perpendicular_implies_a_value (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : let tangent_slope := (Real.log a)
        let perpendicular_line_slope := -1/2
        tangent_slope * perpendicular_line_slope = -1) :
  a = Real.exp 2 := by sorry

end tangent_perpendicular_implies_a_value_l2917_291746


namespace fraction_sum_inequality_l2917_291762

theorem fraction_sum_inequality (α β a b : ℝ) (hα : α > 0) (hβ : β > 0)
  (ha : α ≤ a ∧ a ≤ β) (hb : α ≤ b ∧ b ≤ β) :
  b / a + a / b ≤ β / α + α / β ∧
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β) ∨ (a = β ∧ b = α)) := by
  sorry

end fraction_sum_inequality_l2917_291762


namespace rational_function_uniqueness_l2917_291788

/-- A function from rational numbers to rational numbers -/
def RationalFunction := ℚ → ℚ

/-- The property that f(1) = 2 -/
def HasPropertyOne (f : RationalFunction) : Prop :=
  f 1 = 2

/-- The property that f(xy) = f(x)f(y) - f(x + y) + 1 for all x, y ∈ ℚ -/
def HasPropertyTwo (f : RationalFunction) : Prop :=
  ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1

/-- The theorem stating that any function satisfying both properties must be f(x) = x + 1 -/
theorem rational_function_uniqueness (f : RationalFunction)
  (h1 : HasPropertyOne f) (h2 : HasPropertyTwo f) :
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end rational_function_uniqueness_l2917_291788


namespace planes_perpendicular_l2917_291718

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem statement
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_parallel_α : line_parallel_plane m α)
  (h_n_perp_β : line_perpendicular_plane n β)
  (h_m_parallel_n : parallel m n) :
  plane_perpendicular α β :=
sorry

end planes_perpendicular_l2917_291718


namespace unique_angle_with_same_tangent_l2917_291769

theorem unique_angle_with_same_tangent :
  ∃! (n : ℤ), -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) ∧ n = 150 := by
  sorry

end unique_angle_with_same_tangent_l2917_291769


namespace isosceles_obtuse_triangle_smallest_angle_l2917_291785

/-- 
Given an isosceles, obtuse triangle where the largest angle is 20% larger than 60 degrees,
prove that the measure of each of the two smallest angles is 54 degrees.
-/
theorem isosceles_obtuse_triangle_smallest_angle 
  (α β γ : ℝ) 
  (isosceles : α = β)
  (obtuse : γ > 90)
  (largest_angle : γ = 60 * (1 + 0.2))
  (angle_sum : α + β + γ = 180) :
  α = 54 := by
sorry

end isosceles_obtuse_triangle_smallest_angle_l2917_291785


namespace solution_set_a_2_range_of_a_l2917_291766

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f 2 x < 4} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 7/2} := by sorry

-- Part 2: Range of a when f(x) ≥ 2 for all x
theorem range_of_a :
  (∀ x, f a x ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 := by sorry

end solution_set_a_2_range_of_a_l2917_291766


namespace max_value_theorem_l2917_291740

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 2) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (a + b) / (a * b * c) ≤ (x + y) / (x * y * z)) →
  (x + y) / (x * y * z) = 13.5 :=
sorry

end max_value_theorem_l2917_291740


namespace point_in_first_quadrant_l2917_291774

/-- A proportional function where y increases as x increases -/
structure IncreasingProportionalFunction where
  k : ℝ
  increasing : ∀ x₁ x₂, x₁ < x₂ → k * x₁ < k * x₂

/-- The point (√3, k) is in the first quadrant for an increasing proportional function -/
theorem point_in_first_quadrant (f : IncreasingProportionalFunction) :
  f.k > 0 ∧ Real.sqrt 3 > 0 :=
by sorry

end point_in_first_quadrant_l2917_291774


namespace max_red_squares_is_twelve_l2917_291790

/-- A configuration of colored squares on a 5x5 grid -/
def ColorConfiguration := Fin 5 → Fin 5 → Bool

/-- Checks if four points form an axis-parallel rectangle -/
def isAxisParallelRectangle (p1 p2 p3 p4 : Fin 5 × Fin 5) : Bool :=
  sorry

/-- Checks if a configuration contains an axis-parallel rectangle formed by red squares -/
def containsAxisParallelRectangle (config : ColorConfiguration) : Bool :=
  sorry

/-- Counts the number of red squares in a configuration -/
def countRedSquares (config : ColorConfiguration) : Nat :=
  sorry

/-- The maximum number of red squares possible without forming an axis-parallel rectangle -/
def maxRedSquares : Nat :=
  sorry

theorem max_red_squares_is_twelve :
  maxRedSquares = 12 :=
sorry

end max_red_squares_is_twelve_l2917_291790


namespace rope_for_first_post_l2917_291708

theorem rope_for_first_post (second_post third_post fourth_post total : ℕ) 
  (h1 : second_post = 20)
  (h2 : third_post = 14)
  (h3 : fourth_post = 12)
  (h4 : total = 70)
  (h5 : ∃ first_post : ℕ, first_post + second_post + third_post + fourth_post = total) :
  ∃ first_post : ℕ, first_post = 24 ∧ first_post + second_post + third_post + fourth_post = total :=
by
  sorry

end rope_for_first_post_l2917_291708


namespace joan_driving_speed_l2917_291772

theorem joan_driving_speed 
  (total_distance : ℝ) 
  (total_trip_time : ℝ) 
  (lunch_break : ℝ) 
  (bathroom_break : ℝ) 
  (num_bathroom_breaks : ℕ) :
  total_distance = 480 →
  total_trip_time = 9 →
  lunch_break = 0.5 →
  bathroom_break = 0.25 →
  num_bathroom_breaks = 2 →
  let total_break_time := lunch_break + num_bathroom_breaks * bathroom_break
  let driving_time := total_trip_time - total_break_time
  let speed := total_distance / driving_time
  speed = 60 := by sorry

end joan_driving_speed_l2917_291772


namespace quadratic_solution_implies_value_l2917_291770

theorem quadratic_solution_implies_value (a : ℝ) :
  (2^2 - 3*2 + a = 0) → (2*a - 1 = 3) := by
  sorry

end quadratic_solution_implies_value_l2917_291770


namespace painted_cells_theorem_l2917_291780

theorem painted_cells_theorem (k l : ℕ) : 
  k * l = 74 →
  ((2 * k + 1) * (2 * l + 1) - k * l = 373) ∨
  ((2 * k + 1) * (2 * l + 1) - k * l = 301) := by
sorry

end painted_cells_theorem_l2917_291780


namespace largest_coin_distribution_exists_largest_distribution_l2917_291744

theorem largest_coin_distribution (n : ℕ) : n < 150 → n % 15 = 3 → n ≤ 138 := by
  sorry

theorem exists_largest_distribution : ∃ n : ℕ, n < 150 ∧ n % 15 = 3 ∧ n = 138 := by
  sorry

end largest_coin_distribution_exists_largest_distribution_l2917_291744


namespace a_gt_6_sufficient_not_necessary_for_a_sq_gt_36_l2917_291716

theorem a_gt_6_sufficient_not_necessary_for_a_sq_gt_36 :
  (∀ a : ℝ, a > 6 → a^2 > 36) ∧
  (∃ a : ℝ, a^2 > 36 ∧ a ≤ 6) :=
by sorry

end a_gt_6_sufficient_not_necessary_for_a_sq_gt_36_l2917_291716


namespace only_solution_l2917_291722

/-- Represents the position of a person in a line of recruits. -/
structure Position :=
  (ahead : ℕ)

/-- Represents the three brothers in the line of recruits. -/
inductive Brother
| Peter
| Nikolay
| Denis

/-- Gets the initial number of people ahead of a given brother. -/
def initial_ahead (b : Brother) : ℕ :=
  match b with
  | Brother.Peter => 50
  | Brother.Nikolay => 100
  | Brother.Denis => 170

/-- Calculates the number of people in front after turning, given the total number of recruits. -/
def after_turn (n : ℕ) (b : Brother) : ℕ :=
  n - (initial_ahead b + 1)

/-- Checks if the condition after turning is satisfied for a given total number of recruits. -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ (b1 b2 : Brother), b1 ≠ b2 ∧ after_turn n b1 = 4 * after_turn n b2

/-- The theorem stating that 211 is the only solution. -/
theorem only_solution :
  ∀ n : ℕ, satisfies_condition n ↔ n = 211 :=
sorry

end only_solution_l2917_291722


namespace chord_addition_theorem_sum_of_squares_theorem_l2917_291700

/-- Represents a circle with chords --/
structure ChordedCircle where
  num_chords : ℕ
  num_regions : ℕ

/-- The result of adding a chord to a circle --/
structure ChordAdditionResult where
  min_regions : ℕ
  max_regions : ℕ

/-- Function to add a chord to a circle --/
def add_chord (circle : ChordedCircle) : ChordAdditionResult :=
  { min_regions := circle.num_regions + 1,
    max_regions := circle.num_regions + circle.num_chords + 1 }

/-- Theorem statement --/
theorem chord_addition_theorem (initial_circle : ChordedCircle) 
  (h1 : initial_circle.num_chords = 4) 
  (h2 : initial_circle.num_regions = 9) : 
  let result := add_chord initial_circle
  result.min_regions = 10 ∧ result.max_regions = 14 := by
  sorry

/-- Corollary: The sum of squares of max and min regions --/
theorem sum_of_squares_theorem (initial_circle : ChordedCircle) 
  (h1 : initial_circle.num_chords = 4) 
  (h2 : initial_circle.num_regions = 9) : 
  let result := add_chord initial_circle
  result.max_regions ^ 2 + result.min_regions ^ 2 = 296 := by
  sorry

end chord_addition_theorem_sum_of_squares_theorem_l2917_291700


namespace twenty_team_tournament_games_l2917_291727

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInTournament (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 20 teams requires 19 games to determine the winner. -/
theorem twenty_team_tournament_games :
  gamesInTournament 20 = 19 := by
  sorry

end twenty_team_tournament_games_l2917_291727


namespace smallest_b_value_l2917_291719

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 8) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  ∀ c : ℕ+, c < b → ¬(∃ d : ℕ+, d - c = 8 ∧ 
    Nat.gcd ((d^3 + c^3) / (d + c)) (d * c) = 16) :=
by sorry

end smallest_b_value_l2917_291719


namespace intersection_A_B_l2917_291748

def A : Set ℝ := {-3, -1, 0, 1, 2, 3, 4}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2, 3} := by
  sorry

end intersection_A_B_l2917_291748


namespace sqrt_product_sqrt_three_times_sqrt_two_l2917_291741

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by sorry

theorem sqrt_three_times_sqrt_two :
  Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by sorry

end sqrt_product_sqrt_three_times_sqrt_two_l2917_291741


namespace value_of_c_l2917_291726

theorem value_of_c (a b c : ℝ) : 
  12 = 0.06 * a → 
  6 = 0.12 * b → 
  c = b / a → 
  c = 0.25 := by
sorry

end value_of_c_l2917_291726


namespace existence_of_special_odd_numbers_l2917_291775

theorem existence_of_special_odd_numbers : ∃ m n : ℕ, 
  Odd m ∧ Odd n ∧ 
  m > 2009 ∧ n > 2009 ∧ 
  (n^2 + 8) % m = 0 ∧ 
  (m^2 + 8) % n = 0 := by
sorry

end existence_of_special_odd_numbers_l2917_291775


namespace unique_fraction_representation_l2917_291787

theorem unique_fraction_representation (n : ℕ+) :
  ∃! (a b : ℚ), a > 0 ∧ b > 0 ∧ 
  ∃ (k m : ℤ), a = k / n ∧ b = m / (n + 1) ∧
  (2 * n + 1 : ℚ) / (n * (n + 1)) = a + b :=
by sorry

end unique_fraction_representation_l2917_291787


namespace k_of_h_10_l2917_291720

def h (x : ℝ) : ℝ := 4 * x - 5

def k (x : ℝ) : ℝ := 2 * x + 6

theorem k_of_h_10 : k (h 10) = 76 := by
  sorry

end k_of_h_10_l2917_291720


namespace donation_distribution_l2917_291730

/-- Proves that donating 80% of $2500 to 8 organizations results in each organization receiving $250 --/
theorem donation_distribution (total_amount : ℝ) (donation_percentage : ℝ) (num_organizations : ℕ) :
  total_amount = 2500 →
  donation_percentage = 0.8 →
  num_organizations = 8 →
  (total_amount * donation_percentage) / num_organizations = 250 := by
sorry

end donation_distribution_l2917_291730


namespace seating_arrangement_count_l2917_291758

-- Define the number of people
def num_people : ℕ := 10

-- Define the number of seats in each row
def seats_per_row : ℕ := 5

-- Define a function to calculate the number of valid seating arrangements
def valid_seating_arrangements (n : ℕ) (s : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

-- State the theorem
theorem seating_arrangement_count :
  valid_seating_arrangements num_people seats_per_row = 518400 :=
sorry

end seating_arrangement_count_l2917_291758


namespace product_xyz_l2917_291764

theorem product_xyz (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) (h3 : z ≠ 0) : x * y * z = -4 := by
  sorry

end product_xyz_l2917_291764


namespace arithmetic_sequence_length_l2917_291750

/-- 
Given an arithmetic sequence with:
- First term a₁ = -48
- Common difference d = 5
- Last term aₙ = 72

Prove that the sequence has 25 terms.
-/
theorem arithmetic_sequence_length : 
  let a₁ : ℤ := -48  -- First term
  let d : ℤ := 5     -- Common difference
  let aₙ : ℤ := 72   -- Last term
  ∃ n : ℕ, n = 25 ∧ aₙ = a₁ + (n - 1) * d :=
sorry

end arithmetic_sequence_length_l2917_291750


namespace min_q_value_l2917_291734

def q (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose 50 2

theorem min_q_value (a : ℕ) :
  (∀ x, 1 ≤ x ∧ x < a → q x < 1/2) ∧ q a ≥ 1/2 → a = 7 :=
sorry

end min_q_value_l2917_291734


namespace parabola_circle_intersection_l2917_291731

/-- Given a parabola y² = 2px (p > 0) and a point A(m, 2√2) on it,
    if a circle centered at A with radius |AF| intersects the y-axis
    with a chord of length 2√7, then m = (2√3)/3 -/
theorem parabola_circle_intersection (p m : ℝ) (hp : p > 0) :
  2 * p * m = 8 →
  let f := (2 / m, 0)
  let r := m + 2 / m
  (r^2 - m^2 = 7) →
  m = (2 * Real.sqrt 3) / 3 := by sorry

end parabola_circle_intersection_l2917_291731


namespace polynomial_roots_l2917_291735

def P (x : ℝ) : ℝ := x^6 - 3*x^5 - 6*x^3 - x + 8

theorem polynomial_roots :
  (∀ x < 0, P x > 0) ∧ (∃ x > 0, P x = 0) :=
sorry

end polynomial_roots_l2917_291735


namespace bernoullis_inequality_l2917_291776

theorem bernoullis_inequality (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : n > 0) :
  (1 + x)^n ≥ 1 + n * x := by
  sorry

end bernoullis_inequality_l2917_291776


namespace circle_max_sum_squares_l2917_291717

theorem circle_max_sum_squares :
  ∀ x y : ℝ, x^2 - 4*x - 4 + y^2 = 0 →
  x^2 + y^2 ≤ 12 + 8 * Real.sqrt 2 :=
by sorry

end circle_max_sum_squares_l2917_291717


namespace rectangle_max_area_l2917_291743

theorem rectangle_max_area (d : ℝ) (h : d > 0) :
  ∀ (w h : ℝ), w > 0 → h > 0 → w^2 + h^2 = d^2 →
  w * h ≤ (d^2) / 2 ∧ (w * h = (d^2) / 2 ↔ w = h) :=
sorry

end rectangle_max_area_l2917_291743


namespace extended_pattern_ratio_l2917_291725

/-- Represents the original square pattern of tiles -/
structure OriginalPattern :=
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Represents the extended pattern after adding two layers -/
structure ExtendedPattern :=
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Function to extend the original pattern with two alternating layers -/
def extend_pattern (original : OriginalPattern) : ExtendedPattern :=
  { black_tiles := original.black_tiles + 24,
    white_tiles := original.white_tiles + 32 }

/-- Theorem stating the ratio of black to white tiles in the extended pattern -/
theorem extended_pattern_ratio 
  (original : OriginalPattern) 
  (h1 : original.black_tiles = 9) 
  (h2 : original.white_tiles = 16) :
  let extended := extend_pattern original
  (extended.black_tiles : ℚ) / extended.white_tiles = 33 / 48 := by
  sorry

#check extended_pattern_ratio

end extended_pattern_ratio_l2917_291725


namespace quadratic_root_theorem_l2917_291709

theorem quadratic_root_theorem (a b c d : ℝ) (h : a ≠ 0) :
  (a * (b - c - d) * 1^2 + b * (c - a + d) * 1 + c * (a - b - d) = 0) →
  ∃ x : ℝ, x ≠ 1 ∧ 
    a * (b - c - d) * x^2 + b * (c - a + d) * x + c * (a - b - d) = 0 ∧
    x = c * (a - b - d) / (a * (b - c - d)) :=
by sorry

end quadratic_root_theorem_l2917_291709


namespace investment_interest_l2917_291703

/-- Calculates the compound interest earned given initial investment, interest rate, and time period. -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- The interest earned on a $2000 investment at 2% annual compound interest after 3 years is $122. -/
theorem investment_interest : 
  ∃ ε > 0, |compound_interest 2000 0.02 3 - 122| < ε :=
by
  sorry


end investment_interest_l2917_291703


namespace reading_percentage_third_night_l2917_291757

/-- Theorem: Reading percentage on the third night
Given:
- A book with 500 pages
- 20% read on the first night
- 20% read on the second night
- 150 pages left after three nights of reading
Prove: The percentage read on the third night is 30%
-/
theorem reading_percentage_third_night
  (total_pages : ℕ)
  (first_night_percentage : ℚ)
  (second_night_percentage : ℚ)
  (pages_left : ℕ)
  (h1 : total_pages = 500)
  (h2 : first_night_percentage = 20 / 100)
  (h3 : second_night_percentage = 20 / 100)
  (h4 : pages_left = 150) :
  let pages_read_first_two_nights := (first_night_percentage + second_night_percentage) * total_pages
  let total_pages_read := total_pages - pages_left
  let pages_read_third_night := total_pages_read - pages_read_first_two_nights
  pages_read_third_night / total_pages = 30 / 100 := by
  sorry

end reading_percentage_third_night_l2917_291757


namespace always_on_iff_odd_l2917_291797

/-- Represents the state of a light bulb -/
inductive BulbState
| On
| Off

/-- Represents a configuration of light bulbs -/
def BulbConfiguration (n : ℕ) := Fin n → BulbState

/-- Function to update the state of bulbs according to the given rule -/
def updateBulbs (n : ℕ) (config : BulbConfiguration n) : BulbConfiguration n :=
  sorry

/-- Predicate to check if a configuration has at least one bulb on -/
def hasOnBulb (n : ℕ) (config : BulbConfiguration n) : Prop :=
  sorry

/-- Theorem stating that there exists a configuration that always has at least one bulb on
    if and only if n is odd -/
theorem always_on_iff_odd (n : ℕ) :
  (∃ (initial : BulbConfiguration n), ∀ (t : ℕ), hasOnBulb n ((updateBulbs n)^[t] initial)) ↔ Odd n :=
sorry

end always_on_iff_odd_l2917_291797


namespace robes_savings_l2917_291742

/-- Calculates Robe's initial savings given the repair costs and remaining savings --/
def initial_savings (repair_fee : ℕ) (corner_light_cost : ℕ) (brake_disk_cost : ℕ) (remaining_savings : ℕ) : ℕ :=
  remaining_savings + repair_fee + corner_light_cost + 2 * brake_disk_cost

theorem robes_savings :
  let repair_fee : ℕ := 10
  let corner_light_cost : ℕ := 2 * repair_fee
  let brake_disk_cost : ℕ := 3 * corner_light_cost
  let remaining_savings : ℕ := 480
  initial_savings repair_fee corner_light_cost brake_disk_cost remaining_savings = 630 := by
  sorry

#eval initial_savings 10 20 60 480

end robes_savings_l2917_291742


namespace f_sum_equals_three_l2917_291701

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_sum_equals_three 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_odd : is_odd_function (fun x ↦ f (x - 1))) 
  (h_f2 : f 2 = 3) : 
  f 5 + f 6 = 3 := by
sorry

end f_sum_equals_three_l2917_291701


namespace isabel_homework_problem_l2917_291782

/-- Given a total number of problems, number of finished problems, and number of remaining pages,
    calculate the number of problems per page, assuming each page has an equal number of problems. -/
def problems_per_page (total : ℕ) (finished : ℕ) (pages : ℕ) : ℕ :=
  (total - finished) / pages

/-- Theorem stating that for the given problem, there are 8 problems per page. -/
theorem isabel_homework_problem :
  problems_per_page 72 32 5 = 8 := by
  sorry

end isabel_homework_problem_l2917_291782


namespace angie_necessities_contribution_l2917_291759

def salary : ℕ := 80
def taxes : ℕ := 20
def leftover : ℕ := 18

theorem angie_necessities_contribution :
  salary - taxes - leftover = 42 := by sorry

end angie_necessities_contribution_l2917_291759


namespace class_average_proof_l2917_291705

/-- Given a class with boys and girls, their average scores, and the ratio of boys to girls,
    prove that the overall class average is 94 points. -/
theorem class_average_proof (boys_avg : ℝ) (girls_avg : ℝ) (ratio : ℝ) :
  boys_avg = 90 →
  girls_avg = 96 →
  ratio = 0.5 →
  (ratio * girls_avg + girls_avg) / (ratio + 1) = 94 := by
  sorry

end class_average_proof_l2917_291705


namespace smallest_k_with_remainder_one_l2917_291795

theorem smallest_k_with_remainder_one : ∃! k : ℕ, 
  k > 1 ∧ 
  k % 13 = 1 ∧ 
  k % 7 = 1 ∧ 
  k % 5 = 1 ∧ 
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 7 = 1 ∧ m % 5 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_k_with_remainder_one_l2917_291795


namespace quadratic_inequality_always_nonnegative_l2917_291755

theorem quadratic_inequality_always_nonnegative (x : ℝ) : x^2 + 3 ≥ 0 := by
  sorry

end quadratic_inequality_always_nonnegative_l2917_291755


namespace root_product_theorem_l2917_291761

theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 1 = 0) → 
  (y₂^5 - y₂^3 + 1 = 0) → 
  (y₃^5 - y₃^3 + 1 = 0) → 
  (y₄^5 - y₄^3 + 1 = 0) → 
  (y₅^5 - y₅^3 + 1 = 0) → 
  ((y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = 22) := by
sorry

end root_product_theorem_l2917_291761


namespace complement_of_A_l2917_291713

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem complement_of_A (x : ℝ) : x ∈ (U \ A) ↔ x < 1 ∨ x > 3 := by
  sorry

end complement_of_A_l2917_291713


namespace sin_squared_plus_sin_minus_two_range_l2917_291781

theorem sin_squared_plus_sin_minus_two_range :
  ∀ x : ℝ, -9/4 ≤ Real.sin x ^ 2 + Real.sin x - 2 ∧
  (∃ x : ℝ, Real.sin x ^ 2 + Real.sin x - 2 = -9/4) ∧
  (∃ x : ℝ, Real.sin x ^ 2 + Real.sin x - 2 = 0) := by
  sorry

end sin_squared_plus_sin_minus_two_range_l2917_291781


namespace quadratic_standard_form_l2917_291728

theorem quadratic_standard_form : 
  ∃ (a b c : ℝ), ∀ x, 5 * x^2 = 6 * x - 8 ↔ a * x^2 + b * x + c = 0 ∧ a = 5 ∧ b = -6 ∧ c = 8 := by
  sorry

end quadratic_standard_form_l2917_291728


namespace factor_x6_minus_x4_minus_x2_plus_1_l2917_291732

theorem factor_x6_minus_x4_minus_x2_plus_1 (x : ℝ) :
  x^6 - x^4 - x^2 + 1 = (x - 1) * (x + 1) * (x^2 + 1) := by
  sorry

end factor_x6_minus_x4_minus_x2_plus_1_l2917_291732


namespace inverse_sum_theorem_l2917_291753

noncomputable def g (x : ℝ) : ℝ :=
  if x < 15 then x + 5 else 3 * x - 1

theorem inverse_sum_theorem : 
  (Function.invFun g) 10 + (Function.invFun g) 50 = 22 := by
  sorry

end inverse_sum_theorem_l2917_291753


namespace largest_four_digit_perfect_cube_l2917_291779

theorem largest_four_digit_perfect_cube : 
  ∀ n : ℕ, n ≤ 9999 → n ≥ 1000 → (∃ m : ℕ, n = m^3) → n ≤ 9261 :=
by
  sorry

end largest_four_digit_perfect_cube_l2917_291779


namespace sandwich_availability_l2917_291707

/-- Given an initial number of sandwich kinds and a number of sold-out sandwich kinds,
    prove that the current number of available sandwich kinds is their difference. -/
theorem sandwich_availability (initial : ℕ) (sold_out : ℕ) (h : sold_out ≤ initial) :
  initial - sold_out = initial - sold_out :=
by sorry

end sandwich_availability_l2917_291707


namespace probability_even_product_l2917_291711

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def is_even_product (a b : ℕ) : Prop := Even (a * b)

theorem probability_even_product :
  Nat.card {p : S × S | p.1 ≠ p.2 ∧ is_even_product p.1 p.2} / Nat.choose 7 2 = 5 / 7 := by
  sorry

end probability_even_product_l2917_291711


namespace hundred_day_previous_year_is_thursday_l2917_291778

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (year : Year) (day : ℕ) : DayOfWeek :=
  sorry

/-- Checks if a year is a leap year -/
def isLeapYear (year : Year) : Bool :=
  sorry

theorem hundred_day_previous_year_is_thursday 
  (N : Year)
  (h1 : dayOfWeek N 300 = DayOfWeek.Tuesday)
  (h2 : dayOfWeek (Year.mk (N.value + 1)) 200 = DayOfWeek.Tuesday) :
  dayOfWeek (Year.mk (N.value - 1)) 100 = DayOfWeek.Thursday :=
sorry

end hundred_day_previous_year_is_thursday_l2917_291778


namespace not_necessarily_right_triangle_l2917_291754

theorem not_necessarily_right_triangle (A B C : ℝ) (h : A / B = 3 / 4 ∧ B / C = 4 / 5) :
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end not_necessarily_right_triangle_l2917_291754


namespace drilled_solid_surface_area_l2917_291736

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents the drilled solid S -/
structure DrilledSolid where
  cube : Cube
  tunnelStart : Point3D
  tunnelEnd : Point3D

/-- Calculate the surface area of the drilled solid S -/
def surfaceArea (s : DrilledSolid) : ℝ := sorry

/-- The main theorem stating the surface area of the drilled solid -/
theorem drilled_solid_surface_area 
  (e f g h c d b a i j k : Point3D)
  (cube : Cube)
  (s : DrilledSolid)
  (h1 : cube.edgeLength = 10)
  (h2 : e.x = 10 ∧ e.y = 10 ∧ e.z = 10)
  (h3 : i.x = 7 ∧ i.y = 10 ∧ i.z = 10)
  (h4 : j.x = 10 ∧ j.y = 7 ∧ j.z = 10)
  (h5 : k.x = 10 ∧ k.y = 10 ∧ k.z = 7)
  (h6 : s.cube = cube)
  (h7 : s.tunnelStart = i)
  (h8 : s.tunnelEnd = k) :
  surfaceArea s = 582 + 13.5 * Real.sqrt 6 := by sorry

end drilled_solid_surface_area_l2917_291736


namespace certain_number_problem_l2917_291752

theorem certain_number_problem (x : ℝ) : 4 * (3 * x / 5 - 220) = 320 → x = 500 := by
  sorry

end certain_number_problem_l2917_291752


namespace shoes_mode_median_equal_l2917_291714

structure SalesData where
  sizes : List Float
  volumes : List Nat
  total_pairs : Nat

def mode (data : SalesData) : Float :=
  sorry

def median (data : SalesData) : Float :=
  sorry

theorem shoes_mode_median_equal (data : SalesData) :
  data.sizes = [23, 23.5, 24, 24.5, 25] ∧
  data.volumes = [1, 2, 2, 6, 2] ∧
  data.total_pairs = 15 →
  mode data = 24.5 ∧ median data = 24.5 := by
  sorry

end shoes_mode_median_equal_l2917_291714


namespace polynomial_sum_l2917_291733

theorem polynomial_sum (a b c d : ℝ) : 
  (fun x : ℝ => (4*x^2 - 3*x + 2)*(5 - x)) = 
  (fun x : ℝ => a*x^3 + b*x^2 + c*x + d) → 
  5*a + 3*b + 2*c + d = 25 := by
sorry

end polynomial_sum_l2917_291733


namespace quadratic_maximum_l2917_291760

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 10

/-- The point where the maximum occurs -/
def x_max : ℝ := -2

theorem quadratic_maximum :
  ∀ x : ℝ, f x ≤ f x_max :=
sorry

end quadratic_maximum_l2917_291760


namespace sequence_sum_l2917_291763

theorem sequence_sum (S : ℝ) (a b : ℝ) : 
  (S - a) / 100 = 2022 →
  (S - b) / 100 = 2023 →
  (a + b) / 2 = 51 →
  S = 202301 := by
sorry

end sequence_sum_l2917_291763


namespace sequence_20th_term_l2917_291704

/-- Given a sequence {aₙ} where a₁ = 1 and aₙ₊₁ = aₙ + 2 for n ∈ ℕ*, prove that a₂₀ = 39 -/
theorem sequence_20th_term (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2) : 
  a 20 = 39 := by
  sorry

end sequence_20th_term_l2917_291704


namespace inequality_preservation_l2917_291756

theorem inequality_preservation (a b : ℝ) (h : a > b) : 3 * a > 3 * b := by
  sorry

end inequality_preservation_l2917_291756


namespace equal_roots_condition_l2917_291712

theorem equal_roots_condition (m : ℝ) : 
  (∃! x : ℝ, (x * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = (x + 1) / (m + 1)) ↔ 
  (m = -1 ∨ m = -5) :=
by sorry

end equal_roots_condition_l2917_291712


namespace max_candy_leftover_l2917_291783

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
by sorry

end max_candy_leftover_l2917_291783


namespace dave_breaks_two_strings_per_night_l2917_291768

def shows_per_week : ℕ := 6
def total_weeks : ℕ := 12
def total_strings : ℕ := 144

theorem dave_breaks_two_strings_per_night :
  (total_strings : ℚ) / (shows_per_week * total_weeks) = 2 := by
  sorry

end dave_breaks_two_strings_per_night_l2917_291768


namespace perpendicular_to_same_line_are_parallel_l2917_291710

-- Define the concept of a line in a plane
def Line (P : Type) := P → P → Prop

-- Define the concept of a plane
variable {P : Type}

-- Define the perpendicular relation between lines
def Perpendicular (l₁ l₂ : Line P) : Prop := sorry

-- Define the parallel relation between lines
def Parallel (l₁ l₂ : Line P) : Prop := sorry

-- State the theorem
theorem perpendicular_to_same_line_are_parallel 
  (l₁ l₂ l₃ : Line P) 
  (h₁ : Perpendicular l₁ l₃) 
  (h₂ : Perpendicular l₂ l₃) : 
  Parallel l₁ l₂ :=
sorry

end perpendicular_to_same_line_are_parallel_l2917_291710


namespace quadratic_roots_property_l2917_291729

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) →
  (3 * e^2 + 4 * e - 7 = 0) →
  (d - 2) * (e - 2) = 13/3 := by
sorry

end quadratic_roots_property_l2917_291729


namespace quarter_circle_arc_sum_limit_l2917_291796

/-- The limit of the sum of quarter-circle arcs approaches a quarter of the original circle's circumference --/
theorem quarter_circle_arc_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * (D / n) / 4) - π * D / 4| < ε :=
sorry

end quarter_circle_arc_sum_limit_l2917_291796


namespace laura_debt_l2917_291798

/-- Calculates the total amount owed after applying simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the total amount owed after one year is $37.10 -/
theorem laura_debt : 
  let principal : ℝ := 35
  let rate : ℝ := 0.06
  let time : ℝ := 1
  total_amount_owed principal rate time = 37.10 := by
sorry

end laura_debt_l2917_291798


namespace hillarys_deposit_l2917_291738

/-- Hillary's flea market earnings and deposit problem -/
theorem hillarys_deposit (crafts_sold : ℕ) (price_per_craft extra_tip remaining_cash : ℝ) 
  (h1 : crafts_sold = 3)
  (h2 : price_per_craft = 12)
  (h3 : extra_tip = 7)
  (h4 : remaining_cash = 25) :
  let total_earnings := crafts_sold * price_per_craft + extra_tip
  total_earnings - remaining_cash = 18 := by sorry

end hillarys_deposit_l2917_291738


namespace distance_between_cities_l2917_291706

/-- Proves that the distance between two cities is 300 miles given specific travel conditions -/
theorem distance_between_cities (speed_david speed_lewis : ℝ) (meeting_point : ℝ) : 
  speed_david = 50 →
  speed_lewis = 70 →
  meeting_point = 250 →
  ∃ (time : ℝ), 
    time * speed_david = meeting_point ∧
    time * speed_lewis = 2 * 300 - meeting_point →
  300 = 300 := by
  sorry

end distance_between_cities_l2917_291706


namespace red_cube_possible_l2917_291751

/-- Represents a small cube with colored faces -/
structure SmallCube where
  blue_faces : Nat
  red_faces : Nat

/-- Represents the larger cube assembled from small cubes -/
structure LargeCube where
  small_cubes : List SmallCube
  visible_red_faces : Nat

/-- The theorem to be proved -/
theorem red_cube_possible 
  (cubes : List SmallCube) 
  (h1 : cubes.length = 8)
  (h2 : ∀ c ∈ cubes, c.blue_faces + c.red_faces = 6)
  (h3 : (cubes.map SmallCube.blue_faces).sum = 16)
  (h4 : ∃ lc : LargeCube, lc.small_cubes = cubes ∧ lc.visible_red_faces = 8) :
  ∃ lc : LargeCube, lc.small_cubes = cubes ∧ lc.visible_red_faces = 24 := by
  sorry

end red_cube_possible_l2917_291751
