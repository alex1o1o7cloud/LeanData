import Mathlib

namespace sum_of_squares_and_square_of_sum_l3169_316946

theorem sum_of_squares_and_square_of_sum : (3 + 7)^2 + (3^2 + 7^2) = 158 := by sorry

end sum_of_squares_and_square_of_sum_l3169_316946


namespace power_of_power_l3169_316919

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end power_of_power_l3169_316919


namespace nested_radical_sixteen_l3169_316978

theorem nested_radical_sixteen (x : ℝ) : x = Real.sqrt (16 + x) → x = (1 + Real.sqrt 65) / 2 := by
  sorry

end nested_radical_sixteen_l3169_316978


namespace rachel_songs_total_l3169_316989

theorem rachel_songs_total (albums : ℕ) (songs_per_album : ℕ) (h1 : albums = 8) (h2 : songs_per_album = 2) :
  albums * songs_per_album = 16 := by
  sorry

end rachel_songs_total_l3169_316989


namespace complex_equation_solutions_l3169_316914

theorem complex_equation_solutions :
  let f : ℂ → ℂ := λ z => (z^4 - 1) / (z^3 - 3*z + 2)
  ∃! (s : Finset ℂ), s.card = 3 ∧ ∀ z ∈ s, f z = 0 ∧ ∀ w ∉ s, f w ≠ 0 :=
by sorry

end complex_equation_solutions_l3169_316914


namespace probability_third_is_three_l3169_316960

-- Define the set of permutations
def T : Finset (Fin 6 → Fin 6) :=
  (Finset.univ.filter (λ σ : Fin 6 → Fin 6 => Function.Injective σ ∧ σ 0 ≠ 1))

-- Define the probability of the event
def prob_third_is_three (T : Finset (Fin 6 → Fin 6)) : ℚ :=
  (T.filter (λ σ : Fin 6 → Fin 6 => σ 2 = 2)).card / T.card

-- Theorem statement
theorem probability_third_is_three :
  prob_third_is_three T = 1 / 5 := by
  sorry

end probability_third_is_three_l3169_316960


namespace repair_cost_calculation_l3169_316949

/-- Calculates the repair cost of a machine given its purchase price, transportation charges, profit percentage, and selling price. -/
theorem repair_cost_calculation (purchase_price : ℕ) (transportation_charges : ℕ) (profit_percentage : ℕ) (selling_price : ℕ) : 
  purchase_price = 11000 →
  transportation_charges = 1000 →
  profit_percentage = 50 →
  selling_price = 25500 →
  ∃ (repair_cost : ℕ), 
    repair_cost = 5000 ∧
    selling_price = (purchase_price + repair_cost + transportation_charges) * (100 + profit_percentage) / 100 :=
by sorry

end repair_cost_calculation_l3169_316949


namespace randy_money_problem_l3169_316937

def randy_initial_money (randy_received : ℕ) (randy_gave : ℕ) (randy_left : ℕ) : Prop :=
  ∃ (initial : ℕ), initial + randy_received - randy_gave = randy_left

theorem randy_money_problem :
  randy_initial_money 200 1200 2000 → ∃ (initial : ℕ), initial = 3000 := by
  sorry

end randy_money_problem_l3169_316937


namespace two_lines_two_intersections_l3169_316909

/-- The number of intersection points for n lines on a plane -/
def intersection_points (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: If n lines on a plane intersect at exactly 2 points, then n = 2 -/
theorem two_lines_two_intersections (n : ℕ) (h : intersection_points n = 2) : n = 2 := by
  sorry

end two_lines_two_intersections_l3169_316909


namespace end_time_calculation_l3169_316956

-- Define the structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define the problem parameters
def glowInterval : Nat := 17
def startTime : Time := { hours := 1, minutes := 57, seconds := 58 }
def glowCount : Float := 292.29411764705884

-- Define the function to calculate the ending time
def calculateEndTime (start : Time) (interval : Nat) (count : Float) : Time :=
  sorry

-- Theorem statement
theorem end_time_calculation :
  calculateEndTime startTime glowInterval glowCount = { hours := 3, minutes := 20, seconds := 42 } :=
sorry

end end_time_calculation_l3169_316956


namespace circle_outside_square_area_l3169_316912

/-- The area inside a circle with radius 1/2 but outside a square with side length 1, 
    when both shapes share the same center, is equal to π/4 - 1. -/
theorem circle_outside_square_area :
  let square_side : ℝ := 1
  let circle_radius : ℝ := 1/2
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  circle_area - square_area = π/4 - 1 := by
sorry

end circle_outside_square_area_l3169_316912


namespace speedboat_speed_l3169_316915

/-- Proves that the speed of a speedboat crossing a lake is 30 miles per hour,
    given specific conditions about the lake width, sailboat speed, and wait time. -/
theorem speedboat_speed
  (lake_width : ℝ)
  (sailboat_speed : ℝ)
  (wait_time : ℝ)
  (h_lake : lake_width = 60)
  (h_sail : sailboat_speed = 12)
  (h_wait : wait_time = 3)
  : (lake_width / (lake_width / sailboat_speed - wait_time)) = 30 := by
  sorry

end speedboat_speed_l3169_316915


namespace veronica_initial_marbles_l3169_316901

/-- Represents the number of marbles each person has -/
structure Marbles where
  dilan : ℕ
  martha : ℕ
  phillip : ℕ
  veronica : ℕ

/-- The initial distribution of marbles -/
def initial_marbles : Marbles where
  dilan := 14
  martha := 20
  phillip := 19
  veronica := 7  -- We'll prove this is correct

/-- The number of people -/
def num_people : ℕ := 4

/-- The number of marbles each person has after redistribution -/
def marbles_after_redistribution : ℕ := 15

theorem veronica_initial_marbles :
  (initial_marbles.dilan +
   initial_marbles.martha +
   initial_marbles.phillip +
   initial_marbles.veronica) =
  (num_people * marbles_after_redistribution) :=
by sorry

end veronica_initial_marbles_l3169_316901


namespace terms_are_not_like_l3169_316980

/-- Two algebraic terms are considered like terms if they have the same variables raised to the same powers. -/
def are_like_terms (term1 term2 : Type) : Prop := sorry

/-- The first term in the problem -/
def term1 : Type := sorry

/-- The second term in the problem -/
def term2 : Type := sorry

/-- Theorem stating that the two terms are not like terms -/
theorem terms_are_not_like : ¬(are_like_terms term1 term2) := by sorry

end terms_are_not_like_l3169_316980


namespace parallel_vectors_tan_theta_l3169_316954

open Real

theorem parallel_vectors_tan_theta (θ : ℝ) : 
  let a : Fin 2 → ℝ := ![2, sin θ]
  let b : Fin 2 → ℝ := ![1, cos θ]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → tan θ = 2 := by
  sorry

end parallel_vectors_tan_theta_l3169_316954


namespace initial_cookies_count_l3169_316923

/-- The number of cookies initially in the package -/
def initial_cookies : ℕ := sorry

/-- The number of cookies left after eating some -/
def cookies_left : ℕ := 9

/-- The number of cookies eaten -/
def cookies_eaten : ℕ := 9

/-- Theorem stating that the initial number of cookies is 18 -/
theorem initial_cookies_count : initial_cookies = cookies_left + cookies_eaten := by sorry

end initial_cookies_count_l3169_316923


namespace perpendicular_line_to_plane_l3169_316944

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the perpendicular relation between planes and between lines and planes
variable (perp : Plane → Plane → Prop)
variable (perpL : Line → Plane → Prop)

-- Define the planes and lines
variable (α β γ : Plane) (m n : Line)

-- State the theorem
theorem perpendicular_line_to_plane
  (h1 : intersect α γ = m)
  (h2 : perp β α)
  (h3 : perp β γ)
  (h4 : perpL n α)
  (h5 : perpL n β)
  (h6 : perpL m α) :
  perpL m β :=
sorry

end perpendicular_line_to_plane_l3169_316944


namespace distance_A_to_B_is_300_l3169_316929

/-- The distance between two points A and B, given the following conditions:
    - Monkeys travel from A to B
    - A monkey departs from A every 3 minutes
    - It takes a monkey 12 minutes to travel from A to B
    - A rabbit runs from B to A
    - When the rabbit starts, a monkey has just arrived at B
    - The rabbit encounters 5 monkeys on its way to A
    - The rabbit arrives at A just as another monkey leaves A
    - The rabbit's speed is 3 km/h
-/
def distance_A_to_B : ℝ :=
  let monkey_departure_interval : ℝ := 3 -- minutes
  let monkey_travel_time : ℝ := 12 -- minutes
  let encountered_monkeys : ℕ := 5
  let rabbit_speed : ℝ := 3 * 1000 / 60 -- convert 3 km/h to m/min

  -- Define the distance based on the given conditions
  300 -- meters

theorem distance_A_to_B_is_300 :
  distance_A_to_B = 300 := by sorry

end distance_A_to_B_is_300_l3169_316929


namespace number_equality_l3169_316947

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (25/216) * (1/x)) : x = 144/25 := by
  sorry

end number_equality_l3169_316947


namespace expression_value_l3169_316974

theorem expression_value (a b c d x y : ℤ) :
  (a + b = 0) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (abs x = 3) →  -- absolute value of x is 3
  (y = -1) →     -- y is the largest negative integer
  (2*x - c*d + 6*(a + b) - abs y = 4) ∨ (2*x - c*d + 6*(a + b) - abs y = -8) :=
by sorry

end expression_value_l3169_316974


namespace time_at_6_oclock_l3169_316988

/-- Represents a clock with ticks at each hour -/
structure Clock where
  /-- The time between each tick (in seconds) -/
  tick_interval : ℝ
  /-- The total time for all ticks at 12 o'clock (in seconds) -/
  total_time_at_12 : ℝ

/-- Calculates the time between first and last ticks for a given hour -/
def time_between_ticks (c : Clock) (hour : ℕ) : ℝ :=
  c.tick_interval * (hour - 1)

/-- Theorem stating the time between first and last ticks at 6 o'clock -/
theorem time_at_6_oclock (c : Clock) 
  (h1 : c.total_time_at_12 = 66)
  (h2 : c.tick_interval = c.total_time_at_12 / 11) :
  time_between_ticks c 6 = 30 := by
  sorry

end time_at_6_oclock_l3169_316988


namespace sphere_surface_area_circumscribing_unit_cube_l3169_316936

/-- The surface area of a sphere that circumscribes a cube with edge length 1 is 3π. -/
theorem sphere_surface_area_circumscribing_unit_cube (π : ℝ) : 
  (∃ (S : ℝ), S = 3 * π ∧ 
    S = 4 * π * (((1 : ℝ)^2 + (1 : ℝ)^2 + (1 : ℝ)^2).sqrt / 2)^2) :=
by sorry


end sphere_surface_area_circumscribing_unit_cube_l3169_316936


namespace pyramid_has_one_base_l3169_316924

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a point (apex) --/
structure Pyramid where
  base : Set Point
  apex : Point
  faces : Set (Set Point)

/-- Any pyramid has only one base --/
theorem pyramid_has_one_base (p : Pyramid) : ∃! b : Set Point, b = p.base := by
  sorry

end pyramid_has_one_base_l3169_316924


namespace all_sides_equal_not_imply_rectangle_l3169_316943

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : ℝ)  -- Side lengths
  (α β γ δ : ℝ)  -- Internal angles

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  q.α = q.β ∧ q.β = q.γ ∧ q.γ = q.δ ∧ q.δ = 90

-- Define a quadrilateral with all sides equal
def all_sides_equal (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

-- Theorem statement
theorem all_sides_equal_not_imply_rectangle :
  ∃ q : Quadrilateral, all_sides_equal q ∧ ¬(is_rectangle q) := by
  sorry


end all_sides_equal_not_imply_rectangle_l3169_316943


namespace set_A_is_open_interval_zero_two_l3169_316975

-- Define the function f(x) = x^3 - x
def f (x : ℝ) : ℝ := x^3 - x

-- Define the set A
def A : Set ℝ := {a : ℝ | a > 0 ∧ ∃ x : ℝ, f (x + a) = f x}

-- Theorem statement
theorem set_A_is_open_interval_zero_two :
  A = Set.Ioo 0 2 ∪ {2} :=
sorry

end set_A_is_open_interval_zero_two_l3169_316975


namespace students_not_reading_l3169_316994

theorem students_not_reading (total : ℕ) (three_or_more : ℚ) (two : ℚ) (one : ℚ) :
  total = 240 →
  three_or_more = 1 / 6 →
  two = 35 / 100 →
  one = 5 / 12 →
  ↑total - (↑total * (three_or_more + two + one)) = 16 := by
  sorry

end students_not_reading_l3169_316994


namespace disputed_food_weight_l3169_316968

/-- 
Given a piece of food disputed by a dog and a cat:
- x is the total weight of the piece
- d is the difference in the amount the dog wants to take compared to the cat
- The cat takes (x - d) grams
- The dog takes (x + d) grams
- We know that (x - d) = 300 and (x + d) = 500

This theorem proves that the total weight of the disputed piece is 400 grams.
-/
theorem disputed_food_weight (x d : ℝ) 
  (h1 : x - d = 300) 
  (h2 : x + d = 500) : 
  x = 400 := by
sorry


end disputed_food_weight_l3169_316968


namespace cost_of_tomato_seeds_l3169_316942

theorem cost_of_tomato_seeds :
  let pumpkin_cost : ℚ := 5/2
  let chili_cost : ℚ := 9/10
  let pumpkin_packets : ℕ := 3
  let tomato_packets : ℕ := 4
  let chili_packets : ℕ := 5
  let total_spent : ℚ := 18
  ∃ tomato_cost : ℚ, 
    tomato_cost = 3/2 ∧
    pumpkin_cost * pumpkin_packets + tomato_cost * tomato_packets + chili_cost * chili_packets = total_spent :=
by
  sorry

end cost_of_tomato_seeds_l3169_316942


namespace product_inequality_l3169_316964

theorem product_inequality : 
  (190 * 80 = 19 * 800) → 
  (190 * 80 = 19 * 8 * 100) → 
  (19 * 8 * 10 ≠ 190 * 80) := by
sorry

end product_inequality_l3169_316964


namespace orange_price_is_60_l3169_316950

/-- The price of an orange in cents, given the conditions of the fruit stand problem -/
def orange_price : ℕ :=
  let apple_price : ℕ := 40
  let total_fruits : ℕ := 15
  let initial_avg_price : ℕ := 48
  let final_avg_price : ℕ := 45
  let removed_oranges : ℕ := 3
  60

/-- Theorem stating that the price of an orange is 60 cents -/
theorem orange_price_is_60 :
  orange_price = 60 := by sorry

end orange_price_is_60_l3169_316950


namespace abhay_speed_l3169_316979

theorem abhay_speed (distance : ℝ) (a s : ℝ) : 
  distance = 18 →
  distance / a = distance / s + 2 →
  distance / (2 * a) = distance / s - 1 →
  a = 81 / 10 := by
  sorry

end abhay_speed_l3169_316979


namespace perpendicular_circle_exists_l3169_316911

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define perpendicularity between circles
def isPerpendicular (c1 c2 : Circle) : Prop := sorry

-- Define a point passing through a circle
def passesThroughPoint (c : Circle) (p : ℝ × ℝ) : Prop := sorry

theorem perpendicular_circle_exists (A : ℝ × ℝ) (S1 S2 : Circle) :
  ∃! C : Circle, passesThroughPoint C A ∧ isPerpendicular C S1 ∧ isPerpendicular C S2 := by
  sorry

end perpendicular_circle_exists_l3169_316911


namespace divisible_by_four_or_seven_count_divisible_by_four_or_seven_l3169_316927

theorem divisible_by_four_or_seven (n : Nat) : 
  (∃ k : Nat, n = 4 * k ∨ n = 7 * k) ↔ n ∈ Finset.filter (λ x : Nat => x % 4 = 0 ∨ x % 7 = 0) (Finset.range 61) :=
sorry

theorem count_divisible_by_four_or_seven : 
  (Finset.filter (λ x : Nat => x % 4 = 0 ∨ x % 7 = 0) (Finset.range 61)).card = 21 :=
sorry

end divisible_by_four_or_seven_count_divisible_by_four_or_seven_l3169_316927


namespace swap_values_l3169_316997

theorem swap_values (a b : ℕ) : 
  let c := b
  let b' := a
  let a' := c
  (a' = b ∧ b' = a) :=
by
  sorry

end swap_values_l3169_316997


namespace symmetric_complex_sum_third_quadrant_l3169_316962

/-- Given two complex numbers symmetric with respect to the imaginary axis,
    prove that their sum with one divided by its modulus squared is in the third quadrant -/
theorem symmetric_complex_sum_third_quadrant (z₁ z : ℂ) : 
  z₁ = 2 - I →
  z = -Complex.re z₁ + Complex.im z₁ * I → 
  let w := z₁ / Complex.normSq z₁ + z
  Complex.re w < 0 ∧ Complex.im w < 0 := by
  sorry

end symmetric_complex_sum_third_quadrant_l3169_316962


namespace paper_fold_crease_length_l3169_316918

theorem paper_fold_crease_length :
  ∀ (width : ℝ) (angle : ℝ),
  width = 8 →
  angle = π / 4 →
  ∃ (crease_length : ℝ),
  crease_length = 4 * Real.sqrt 2 :=
by sorry

end paper_fold_crease_length_l3169_316918


namespace convex_polyhedron_same_edge_count_l3169_316917

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : ℕ
  max_edges : ℕ
  faces_ge_max_edges : faces ≥ max_edges
  min_edges_per_face : max_edges ≥ 3

/-- Theorem: A convex polyhedron always has two faces with the same number of edges -/
theorem convex_polyhedron_same_edge_count (P : ConvexPolyhedron) :
  ∃ (e : ℕ) (f₁ f₂ : ℕ), f₁ ≠ f₂ ∧ f₁ ≤ P.faces ∧ f₂ ≤ P.faces ∧
  (∃ (edges_of_face : ℕ → ℕ), 
    (∀ f, f ≤ P.faces → 3 ≤ edges_of_face f ∧ edges_of_face f ≤ P.max_edges) ∧
    edges_of_face f₁ = e ∧ edges_of_face f₂ = e) :=
sorry

end convex_polyhedron_same_edge_count_l3169_316917


namespace ball_max_height_l3169_316965

/-- The height of the ball as a function of time -/
def f (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 45 := by
  sorry

end ball_max_height_l3169_316965


namespace factorial_ratio_l3169_316913

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end factorial_ratio_l3169_316913


namespace expression_value_l3169_316910

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + x*y*z = -7 := by
sorry

end expression_value_l3169_316910


namespace infinite_image_is_infinite_l3169_316982

-- Define the concept of an infinite set
def IsInfinite (α : Type*) : Prop := ∃ f : α → α, Function.Injective f ∧ ¬Function.Surjective f

-- State the theorem
theorem infinite_image_is_infinite {A B : Type*} (f : A → B) (h : IsInfinite A) : IsInfinite B := by
  sorry

end infinite_image_is_infinite_l3169_316982


namespace jason_age_2004_l3169_316977

/-- Jason's age at the end of 1997 -/
def jason_age_1997 : ℝ := 35.5

/-- Jason's grandmother's age at the end of 1997 -/
def grandmother_age_1997 : ℝ := 3 * jason_age_1997

/-- The sum of Jason's and his grandmother's birth years -/
def birth_years_sum : ℕ := 3852

/-- The year we're considering for Jason's age -/
def target_year : ℕ := 2004

/-- The reference year for ages -/
def reference_year : ℕ := 1997

theorem jason_age_2004 :
  jason_age_1997 + (target_year - reference_year) = 42.5 ∧
  jason_age_1997 = grandmother_age_1997 / 3 ∧
  (reference_year - jason_age_1997) + (reference_year - grandmother_age_1997) = birth_years_sum :=
by sorry

end jason_age_2004_l3169_316977


namespace one_fourth_divided_by_one_eighth_l3169_316981

theorem one_fourth_divided_by_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end one_fourth_divided_by_one_eighth_l3169_316981


namespace fifth_root_fraction_l3169_316908

theorem fifth_root_fraction : 
  (9 / 16.2) ^ (1/5 : ℝ) = (5/9 : ℝ) ^ (1/5 : ℝ) := by sorry

end fifth_root_fraction_l3169_316908


namespace min_value_theorem_range_theorem_l3169_316939

-- Define the variables and conditions
variable (a b : ℝ) (hsum : a + b = 1) (ha : a > 0) (hb : b > 0)

-- Part I: Minimum value theorem
theorem min_value_theorem : 
  ∃ (min : ℝ), min = 9 ∧ ∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → 1/a + 4/b ≥ min :=
sorry

-- Part II: Range theorem
theorem range_theorem :
  ∀ (x : ℝ), (∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → 1/a + 4/b ≥ |2*x - 1| - |x + 1|) ↔ x ∈ Set.Icc (-7) 11 :=
sorry

end min_value_theorem_range_theorem_l3169_316939


namespace cyclic_sum_inequality_l3169_316990

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) ≥ 2/3 := by
  sorry

end cyclic_sum_inequality_l3169_316990


namespace more_selected_in_B_l3169_316904

def total_candidates : ℕ := 8000
def selection_rate_A : ℚ := 6 / 100
def selection_rate_B : ℚ := 7 / 100

theorem more_selected_in_B : 
  ⌊(selection_rate_B * total_candidates : ℚ)⌋ - ⌊(selection_rate_A * total_candidates : ℚ)⌋ = 80 := by
  sorry

end more_selected_in_B_l3169_316904


namespace gravel_path_cost_is_360_l3169_316996

/-- Calculates the cost of gravelling a path around a rectangular plot -/
def gravel_path_cost (plot_length plot_width path_width : ℝ) (cost_per_sqm : ℝ) : ℝ :=
  let outer_length := plot_length + 2 * path_width
  let outer_width := plot_width + 2 * path_width
  let path_area := outer_length * outer_width - plot_length * plot_width
  path_area * cost_per_sqm

/-- Theorem: The cost of gravelling the path is 360 rupees -/
theorem gravel_path_cost_is_360 :
  gravel_path_cost 110 65 2.5 0.4 = 360 := by
  sorry

end gravel_path_cost_is_360_l3169_316996


namespace nickel_difference_is_zero_l3169_316940

/-- Represents the number of coins of each type -/
structure CoinCollection where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins -/
def total_coins : ℕ := 150

/-- The total value of the coins in cents -/
def total_value : ℕ := 2000

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Checks if a coin collection is valid -/
def is_valid_collection (c : CoinCollection) : Prop :=
  c.nickels + c.dimes + c.quarters = total_coins ∧
  c.nickels * nickel_value + c.dimes * dime_value + c.quarters * quarter_value = total_value ∧
  c.nickels > 0 ∧ c.dimes > 0 ∧ c.quarters > 0

/-- The theorem to be proved -/
theorem nickel_difference_is_zero :
  ∃ (min_nickels max_nickels : ℕ),
    (∀ c : CoinCollection, is_valid_collection c → c.nickels ≥ min_nickels) ∧
    (∀ c : CoinCollection, is_valid_collection c → c.nickels ≤ max_nickels) ∧
    max_nickels - min_nickels = 0 := by
  sorry

end nickel_difference_is_zero_l3169_316940


namespace fruits_left_l3169_316905

-- Define the initial quantities of fruits
def initial_bananas : ℕ := 12
def initial_apples : ℕ := 7
def initial_grapes : ℕ := 19

-- Define the quantities of fruits eaten
def eaten_bananas : ℕ := 4
def eaten_apples : ℕ := 2
def eaten_grapes : ℕ := 10

-- Define the function to calculate remaining fruits
def remaining_fruits : ℕ := 
  (initial_bananas - eaten_bananas) + 
  (initial_apples - eaten_apples) + 
  (initial_grapes - eaten_grapes)

-- Theorem statement
theorem fruits_left : remaining_fruits = 22 := by
  sorry

end fruits_left_l3169_316905


namespace polynomial_equality_sum_l3169_316921

theorem polynomial_equality_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = 
    a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + 
    a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + 
    a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 := by
sorry

end polynomial_equality_sum_l3169_316921


namespace p_true_and_q_false_l3169_316955

-- Define proposition P
def P : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 1/x₀ > 3

-- Define proposition q
def q : Prop := ∀ x : ℝ, x > 2 → x^2 > 2^x

-- Theorem statement
theorem p_true_and_q_false : P ∧ ¬q := by
  sorry

end p_true_and_q_false_l3169_316955


namespace inequality_proof_l3169_316998

theorem inequality_proof (x : ℝ) : x^2 + 1 + 1/(x^2 + 1) ≥ 2 := by
  sorry

end inequality_proof_l3169_316998


namespace power_division_l3169_316922

theorem power_division (x : ℝ) : x^8 / x^2 = x^6 := by
  sorry

end power_division_l3169_316922


namespace nursery_school_age_distribution_l3169_316971

theorem nursery_school_age_distribution (total : ℕ) (four_and_older : ℕ) (not_between_three_and_four : ℕ) :
  total = 50 →
  four_and_older = total / 10 →
  not_between_three_and_four = 25 →
  four_and_older + (total - four_and_older - (total - not_between_three_and_four)) = not_between_three_and_four →
  total - four_and_older - (total - not_between_three_and_four) = 20 := by
sorry

end nursery_school_age_distribution_l3169_316971


namespace union_of_A_and_B_l3169_316920

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| < 2}
def B : Set ℝ := {x : ℝ | x^2 - 3*x < 0}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end union_of_A_and_B_l3169_316920


namespace parallel_vectors_l3169_316906

/-- Given vectors a and b in ℝ², prove that k = -1/3 makes k*a + b parallel to a - 3*b -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-3, 2)) :
  let k : ℝ := -1/3
  let v1 : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
  let v2 : ℝ × ℝ := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  ∃ (c : ℝ), v1 = (c * v2.1, c * v2.2) := by sorry

end parallel_vectors_l3169_316906


namespace inscribed_octagon_area_l3169_316952

theorem inscribed_octagon_area (r : ℝ) (h : r^2 * Real.pi = 400 * Real.pi) :
  2 * r^2 * (1 + Real.sqrt 2) = 800 + 800 * Real.sqrt 2 := by
  sorry

end inscribed_octagon_area_l3169_316952


namespace angle_problem_l3169_316928

theorem angle_problem (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (h3 : angle1 = 80)
  (h4 : angle2 = 100) :
  angle4 = 40 := by
  sorry

end angle_problem_l3169_316928


namespace sine_integral_negative_l3169_316945

theorem sine_integral_negative : ∫ x in -Real.pi..0, Real.sin x < 0 := by
  sorry

end sine_integral_negative_l3169_316945


namespace domain_of_f_l3169_316938

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 - 2*x + 1)^(1/3) + (9 - x^2)^(1/3)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
sorry

end domain_of_f_l3169_316938


namespace road_length_probability_l3169_316957

/-- The probability of a road from A to B being at least 5 miles long -/
def prob_ab : ℚ := 2/3

/-- The probability of a road from B to C being at least 5 miles long -/
def prob_bc : ℚ := 3/4

/-- The probability that at least one of two randomly picked roads
    (one from A to B, one from B to C) is at least 5 miles long -/
def prob_at_least_one : ℚ := 1 - (1 - prob_ab) * (1 - prob_bc)

theorem road_length_probability : prob_at_least_one = 11/12 := by
  sorry

end road_length_probability_l3169_316957


namespace determinant_property_l3169_316935

theorem determinant_property (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 4 →
  Matrix.det ![![a + 2*c, b + 2*d], ![c, d]] = 4 := by
  sorry

end determinant_property_l3169_316935


namespace max_blocks_fit_l3169_316992

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the problem of fitting smaller blocks into a larger box -/
structure BlockFittingProblem where
  largeBox : BoxDimensions
  smallBlock : BoxDimensions

/-- Calculates the maximum number of blocks that can fit based on volume -/
def maxBlocksByVolume (p : BlockFittingProblem) : ℕ :=
  (boxVolume p.largeBox) / (boxVolume p.smallBlock)

/-- Calculates the maximum number of blocks that can fit based on physical arrangement -/
def maxBlocksByArrangement (p : BlockFittingProblem) : ℕ :=
  (p.largeBox.length / p.smallBlock.length) *
  (p.largeBox.width / p.smallBlock.width) *
  (p.largeBox.height / p.smallBlock.height)

/-- The main theorem stating that the maximum number of blocks that can fit is 6 -/
theorem max_blocks_fit (p : BlockFittingProblem) 
    (h1 : p.largeBox = ⟨4, 3, 2⟩) 
    (h2 : p.smallBlock = ⟨3, 1, 1⟩) : 
    min (maxBlocksByVolume p) (maxBlocksByArrangement p) = 6 := by
  sorry


end max_blocks_fit_l3169_316992


namespace hexagon_count_l3169_316932

/-- Represents a regular hexagon divided into smaller equilateral triangles -/
structure DividedHexagon where
  side_length : ℕ
  num_small_triangles : ℕ
  small_triangle_side : ℕ

/-- Counts the number of regular hexagons that can be formed in a divided hexagon -/
def count_hexagons (h : DividedHexagon) : ℕ :=
  sorry

/-- Theorem stating the number of hexagons in the specific configuration -/
theorem hexagon_count (h : DividedHexagon) 
  (h_side : h.side_length = 3)
  (h_triangles : h.num_small_triangles = 54)
  (h_small_side : h.small_triangle_side = 1) :
  count_hexagons h = 36 :=
sorry

end hexagon_count_l3169_316932


namespace circle_radius_l3169_316966

theorem circle_radius (A C : ℝ) (h : A / C = 25) : 
  ∃ r : ℝ, r > 0 ∧ A = π * r^2 ∧ C = 2 * π * r ∧ r = 50 := by
  sorry

end circle_radius_l3169_316966


namespace two_digit_number_problem_l3169_316930

/-- 
Given a two-digit number n = 10a + b, where a and b are single digits,
if 1000a + 100b = 37(100a + 10b + 1), then n = 27.
-/
theorem two_digit_number_problem (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : 1000 * a + 100 * b = 37 * (100 * a + 10 * b + 1)) :
  10 * a + b = 27 := by
  sorry

#check two_digit_number_problem

end two_digit_number_problem_l3169_316930


namespace fourth_root_256_times_cube_root_64_times_sqrt_16_l3169_316993

theorem fourth_root_256_times_cube_root_64_times_sqrt_16 :
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 64 :=
by sorry

end fourth_root_256_times_cube_root_64_times_sqrt_16_l3169_316993


namespace average_percentage_increase_l3169_316972

/-- Given an item with original price of 100 yuan, increased first by 40% and then by 10%,
    prove that the average percentage increase x per time satisfies (1 + 40%)(1 + 10%) = (1 + x)² -/
theorem average_percentage_increase (original_price : ℝ) (first_increase second_increase : ℝ) 
  (x : ℝ) (h1 : original_price = 100) (h2 : first_increase = 0.4) (h3 : second_increase = 0.1) :
  (1 + first_increase) * (1 + second_increase) = (1 + x)^2 := by
  sorry

end average_percentage_increase_l3169_316972


namespace triangle_area_l3169_316970

/-- The area of a triangle with vertices A(0,0), B(1424233,2848467), and C(1424234,2848469) is 1/2 -/
theorem triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1424233, 2848467)
  let C : ℝ × ℝ := (1424234, 2848469)
  let triangle_area := abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2
  triangle_area = 1/2 := by
sorry

#eval (1424233 * 2848469 - 1424234 * 2848467) / 2

end triangle_area_l3169_316970


namespace quadratic_circle_intersection_l3169_316916

/-- Given a quadratic polynomial ax^2 + bx + c where a ≠ 0, if a circle passes through
    its three intersection points with the coordinate axes and intersects the y-axis
    at a fourth point with ordinate y₀, then y₀ = 1/a -/
theorem quadratic_circle_intersection 
  (a b c : ℝ) (h : a ≠ 0) : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
    (∃ y₀ : ℝ, y₀ * c = x₁ * x₂) →
    (∀ y₀ : ℝ, y₀ * c = x₁ * x₂ → y₀ = 1 / a) :=
by sorry

end quadratic_circle_intersection_l3169_316916


namespace triangle_theorem_l3169_316953

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = Real.sqrt 3 * t.a * t.c)
  (h2 : 2 * t.b * Real.cos t.A = Real.sqrt 3 * (t.c * Real.cos t.A + t.a * Real.cos t.C))
  (h3 : (t.a^2 + t.b^2 + t.c^2 - (t.b^2 + t.c^2 - t.a^2) / 2) / 4 = 7) :
  t.B = π / 6 ∧ t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 := by
  sorry

end triangle_theorem_l3169_316953


namespace tunnel_length_l3169_316999

/-- The length of a tunnel given train and time information --/
theorem tunnel_length (train_length : ℝ) (train_speed : ℝ) (total_time : ℝ) (front_exit_time : ℝ) :
  train_length = 1 ∧ 
  train_speed = 30 ∧ 
  total_time = 5 ∧ 
  front_exit_time = 3 →
  1 = train_speed * front_exit_time - train_length / 2 := by
  sorry

#check tunnel_length

end tunnel_length_l3169_316999


namespace division_problem_l3169_316902

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) : 
  dividend = divisor + 2016 →
  quotient = 15 →
  dividend = divisor * quotient →
  dividend = 2160 :=
by
  sorry

end division_problem_l3169_316902


namespace grand_hall_expenditure_l3169_316973

/-- Calculates the total expenditure for covering a rectangular floor with a mat -/
def total_expenditure (length width cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Proves that the total expenditure for covering a 50m × 30m floor with a mat 
    costing Rs. 100 per square meter is Rs. 150,000 -/
theorem grand_hall_expenditure :
  total_expenditure 50 30 100 = 150000 := by
  sorry

end grand_hall_expenditure_l3169_316973


namespace smallest_constant_for_sum_squares_inequality_l3169_316934

theorem smallest_constant_for_sum_squares_inequality :
  ∃ k : ℝ, k > 0 ∧
  (∀ y₁ y₂ y₃ A : ℝ,
    y₁ + y₂ + y₃ = 0 →
    A = max (abs y₁) (max (abs y₂) (abs y₃)) →
    y₁^2 + y₂^2 + y₃^2 ≥ k * A^2) ∧
  (∀ k' : ℝ, k' < k →
    ∃ y₁ y₂ y₃ A : ℝ,
      y₁ + y₂ + y₃ = 0 ∧
      A = max (abs y₁) (max (abs y₂) (abs y₃)) ∧
      y₁^2 + y₂^2 + y₃^2 < k' * A^2) ∧
  k = 1.5 := by
sorry

end smallest_constant_for_sum_squares_inequality_l3169_316934


namespace gcd_problem_l3169_316926

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 1632 * k) : 
  Int.gcd (a^2 + 13*a + 36) (a + 6) = 6 := by
sorry

end gcd_problem_l3169_316926


namespace largest_four_digit_product_l3169_316931

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_four_digit_product (m x y z : ℕ) : 
  m > 0 →
  m = x * y * (10 * x + y) * z →
  is_prime x →
  is_prime y →
  is_prime (10 * x + y) →
  is_prime z →
  x < 20 →
  y < 20 →
  z < 20 →
  x ≠ y →
  x ≠ 10 * x + y →
  y ≠ 10 * x + y →
  x ≠ z →
  y ≠ z →
  (10 * x + y) ≠ z →
  1000 ≤ m →
  m < 10000 →
  m ≤ 7478 :=
sorry

end largest_four_digit_product_l3169_316931


namespace distance_covered_proof_l3169_316987

/-- Calculates the distance covered given a fuel-to-distance ratio and fuel consumption -/
def distance_covered (fuel_ratio : ℚ) (distance_ratio : ℚ) (fuel_consumed : ℚ) : ℚ :=
  (distance_ratio / fuel_ratio) * fuel_consumed

/-- Proves that given a fuel-to-distance ratio of 4:7 and fuel consumption of 44 gallons, 
    the distance covered is 77 miles -/
theorem distance_covered_proof :
  let fuel_ratio : ℚ := 4
  let distance_ratio : ℚ := 7
  let fuel_consumed : ℚ := 44
  distance_covered fuel_ratio distance_ratio fuel_consumed = 77 := by
sorry

end distance_covered_proof_l3169_316987


namespace chocolate_doughnut_cost_l3169_316933

/-- The cost of a chocolate doughnut given the number of students wanting each type,
    the cost of glazed doughnuts, and the total cost. -/
theorem chocolate_doughnut_cost
  (chocolate_students : ℕ)
  (glazed_students : ℕ)
  (glazed_cost : ℚ)
  (total_cost : ℚ)
  (h1 : chocolate_students = 10)
  (h2 : glazed_students = 15)
  (h3 : glazed_cost = 1)
  (h4 : total_cost = 35) :
  ∃ (chocolate_cost : ℚ),
    chocolate_cost * chocolate_students + glazed_cost * glazed_students = total_cost ∧
    chocolate_cost = 2 := by
  sorry

end chocolate_doughnut_cost_l3169_316933


namespace friends_recycled_sixteen_pounds_l3169_316985

/-- Represents the recycling scenario -/
structure RecyclingScenario where
  pounds_per_point : ℕ
  vanessa_pounds : ℕ
  total_points : ℕ

/-- Calculates the amount of paper recycled by Vanessa's friends -/
def friends_recycled_pounds (scenario : RecyclingScenario) : ℕ :=
  scenario.total_points * scenario.pounds_per_point - scenario.vanessa_pounds

/-- Theorem stating that Vanessa's friends recycled 16 pounds -/
theorem friends_recycled_sixteen_pounds :
  ∃ (scenario : RecyclingScenario),
    scenario.pounds_per_point = 9 ∧
    scenario.vanessa_pounds = 20 ∧
    scenario.total_points = 4 ∧
    friends_recycled_pounds scenario = 16 := by
  sorry


end friends_recycled_sixteen_pounds_l3169_316985


namespace min_benches_for_equal_seating_l3169_316967

/-- Represents the seating capacity of a bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Defines the standard bench capacity -/
def standard_bench : BenchCapacity := ⟨8, 12⟩

/-- Defines the extended bench capacity -/
def extended_bench : BenchCapacity := ⟨8, 16⟩

/-- Theorem stating the minimum number of benches required -/
theorem min_benches_for_equal_seating :
  ∃ (n : Nat), n > 0 ∧
    n * standard_bench.adults + n * extended_bench.adults =
    n * standard_bench.children + n * extended_bench.children ∧
    ∀ (m : Nat), m > 0 →
      m * standard_bench.adults + m * extended_bench.adults =
      m * standard_bench.children + m * extended_bench.children →
      m ≥ n :=
by sorry

end min_benches_for_equal_seating_l3169_316967


namespace square_root_cube_root_relation_l3169_316991

theorem square_root_cube_root_relation (x : ℝ) : 
  (∃ y : ℝ, y^2 = x ∧ (y = 8 ∨ y = -8)) → x^(1/3) = 4 := by
  sorry

end square_root_cube_root_relation_l3169_316991


namespace tan_two_theta_minus_pi_over_six_l3169_316907

theorem tan_two_theta_minus_pi_over_six (θ : Real) 
  (h : 4 * Real.cos (θ + π/3) * Real.cos (θ - π/6) = Real.sin (2*θ)) : 
  Real.tan (2*θ - π/6) = Real.sqrt 3 / 9 := by
  sorry

end tan_two_theta_minus_pi_over_six_l3169_316907


namespace smallest_b_value_b_equals_one_l3169_316925

def gcd_notation (a b : ℕ) : ℕ := Nat.gcd a b

theorem smallest_b_value (b : ℕ) : b > 0 → (gcd_notation (gcd_notation 16 20) (gcd_notation 18 b) = 2) → b ≥ 1 :=
by
  sorry

theorem b_equals_one : ∃ (b : ℕ), b > 0 ∧ (gcd_notation (gcd_notation 16 20) (gcd_notation 18 b) = 2) ∧ 
  ∀ (c : ℕ), c > 0 → (gcd_notation (gcd_notation 16 20) (gcd_notation 18 c) = 2) → b ≤ c :=
by
  sorry

end smallest_b_value_b_equals_one_l3169_316925


namespace smallest_positive_period_of_f_l3169_316903

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem smallest_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi :=
sorry

end smallest_positive_period_of_f_l3169_316903


namespace circle_equation_l3169_316984

theorem circle_equation 
  (a b : ℝ) 
  (h1 : a > 0 ∧ b > 0)  -- Center in first quadrant
  (h2 : 2 * a - b + 1 = 0)  -- Center on the line 2x - y + 1 = 0
  (h3 : (a + 4)^2 + (b - 3)^2 = 5^2)  -- Passes through (-4, 3) with radius 5
  : ∀ x y : ℝ, (x - 1)^2 + (y - 3)^2 = 25 ↔ (x - a)^2 + (y - b)^2 = 5^2 := by
sorry

end circle_equation_l3169_316984


namespace median_of_special_sequence_l3169_316900

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_of_special_sequence : 
  let N : ℕ := sequence_sum 200
  let median_index : ℕ := N / 2
  let cumulative_count (n : ℕ) := sequence_sum n
  ∃ (n : ℕ), 
    cumulative_count n ≥ median_index ∧ 
    cumulative_count (n - 1) < median_index ∧
    n = 141 :=
by sorry

end median_of_special_sequence_l3169_316900


namespace binomial_60_3_l3169_316963

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l3169_316963


namespace divided_square_plot_area_l3169_316941

/-- Represents a rectangular plot -/
structure RectangularPlot where
  length : ℝ
  width : ℝ

/-- Represents a square plot divided into 8 equal rectangular parts -/
structure DividedSquarePlot where
  part : RectangularPlot
  perimeter : ℝ

/-- The perimeter of a rectangular plot -/
def RectangularPlot.perimeter (r : RectangularPlot) : ℝ :=
  2 * (r.length + r.width)

/-- The area of a square plot -/
def square_area (side : ℝ) : ℝ :=
  side * side

theorem divided_square_plot_area (d : DividedSquarePlot) 
    (h1 : d.part.length = 2 * d.part.width)
    (h2 : d.perimeter = d.part.perimeter) :
    square_area (4 * d.part.width) = (4 * d.perimeter^2) / 9 := by
  sorry

end divided_square_plot_area_l3169_316941


namespace sin_cos_sum_fifteen_seventyfive_l3169_316983

theorem sin_cos_sum_fifteen_seventyfive : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (75 * π / 180) = 1 := by
  sorry

end sin_cos_sum_fifteen_seventyfive_l3169_316983


namespace robotics_camp_age_problem_l3169_316959

theorem robotics_camp_age_problem (total_members : ℕ) (overall_avg_age : ℕ) 
  (num_girls num_boys num_adults : ℕ) (avg_age_girls avg_age_boys : ℕ) :
  total_members = 60 →
  overall_avg_age = 20 →
  num_girls = 30 →
  num_boys = 20 →
  num_adults = 10 →
  avg_age_girls = 18 →
  avg_age_boys = 22 →
  num_girls + num_boys + num_adults = total_members →
  (avg_age_girls * num_girls + avg_age_boys * num_boys + 
   22 * num_adults : ℕ) / total_members = overall_avg_age :=
by sorry

end robotics_camp_age_problem_l3169_316959


namespace polynomial_inequality_l3169_316961

theorem polynomial_inequality (x : ℝ) : 
  x^6 + 4*x^5 + 2*x^4 - 6*x^3 - 2*x^2 + 4*x - 1 ≥ 0 ↔ 
  x ≤ -1 - Real.sqrt 2 ∨ x = (-1 - Real.sqrt 5) / 2 ∨ x ≥ -1 + Real.sqrt 2 := by
sorry

end polynomial_inequality_l3169_316961


namespace chef_leftover_potatoes_l3169_316948

/-- Given a chef's potato and fry situation, calculate the number of leftover potatoes. -/
def leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ) : ℕ :=
  total_potatoes - (required_fries / fries_per_potato)

/-- Prove that the chef will have 7 potatoes leftover. -/
theorem chef_leftover_potatoes :
  leftover_potatoes 25 15 200 = 7 := by
  sorry

end chef_leftover_potatoes_l3169_316948


namespace matrix_commute_result_l3169_316995

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]

def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]

theorem matrix_commute_result (a b c d : ℝ) (h1 : A * B a b c d = B a b c d * A) 
  (h2 : 4 * b ≠ c) : (a - d) / (c - 4 * b) = 0 := by
  sorry

end matrix_commute_result_l3169_316995


namespace triangle_properties_l3169_316958

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (c * Real.sin B = Real.sqrt 3 * Real.cos C) →
  (a + b = 6) →
  (C = π / 3 ∧ a + b + c ≥ 9) :=
by sorry

end triangle_properties_l3169_316958


namespace arrangement_count_eq_960_l3169_316976

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a row,
    where the elderly people must be adjacent but not at the ends. -/
def arrangement_count : ℕ :=
  let n_volunteers : ℕ := 5
  let n_elderly : ℕ := 2
  let n_total : ℕ := n_volunteers + n_elderly
  let n_ends : ℕ := 2
  let n_remaining_volunteers : ℕ := n_volunteers - n_ends
  let elderly_group : ℕ := 1  -- Treat adjacent elderly as one group

  (n_volunteers.choose n_ends) *    -- Ways to choose volunteers for the ends
  ((n_remaining_volunteers + elderly_group).factorial) *  -- Ways to arrange middle positions
  (n_elderly.factorial)             -- Ways to arrange within elderly group

theorem arrangement_count_eq_960 : arrangement_count = 960 := by
  sorry

end arrangement_count_eq_960_l3169_316976


namespace sandwich_problem_solution_l3169_316951

/-- Represents the sandwich making problem --/
def sandwich_problem (bread_packages : ℕ) (bread_slices_per_package : ℕ) 
  (ham_packages : ℕ) (ham_slices_per_package : ℕ)
  (turkey_packages : ℕ) (turkey_slices_per_package : ℕ)
  (roast_beef_packages : ℕ) (roast_beef_slices_per_package : ℕ)
  (ham_proportion : ℚ) (turkey_proportion : ℚ) (roast_beef_proportion : ℚ) : Prop :=
  let total_bread := bread_packages * bread_slices_per_package
  let total_ham := ham_packages * ham_slices_per_package
  let total_turkey := turkey_packages * turkey_slices_per_package
  let total_roast_beef := roast_beef_packages * roast_beef_slices_per_package
  let total_sandwiches := min (total_ham / ham_proportion) 
                              (min (total_turkey / turkey_proportion) 
                                   (total_roast_beef / roast_beef_proportion))
  let bread_used := 2 * total_sandwiches
  let leftover_bread := total_bread - bread_used
  leftover_bread = 16

/-- The sandwich problem theorem --/
theorem sandwich_problem_solution : 
  sandwich_problem 4 24 3 14 2 18 1 10 (2/5) (7/20) (1/4) := by
  sorry

end sandwich_problem_solution_l3169_316951


namespace subset_of_A_l3169_316969

def A : Set ℝ := {x | x > -1}

theorem subset_of_A : {0} ⊆ A := by sorry

end subset_of_A_l3169_316969


namespace complex_cube_equation_l3169_316986

def complex (x y : ℤ) := x + y * Complex.I

theorem complex_cube_equation (x y d : ℤ) (hx : x > 0) (hy : y > 0) :
  (complex x y)^3 = complex (-26) d → complex x y = complex 1 3 := by
  sorry

end complex_cube_equation_l3169_316986
