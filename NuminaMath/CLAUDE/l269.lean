import Mathlib

namespace boat_speed_is_twelve_l269_26988

/-- Represents the speed of a boat and current in a river --/
structure RiverJourney where
  boat_speed : ℝ
  current_speed : ℝ

/-- Represents the time taken for upstream and downstream journeys --/
structure JourneyTimes where
  upstream_time : ℝ
  downstream_time : ℝ

/-- Checks if the given boat speed is consistent with the journey times --/
def is_consistent_speed (journey : RiverJourney) (times : JourneyTimes) : Prop :=
  (journey.boat_speed - journey.current_speed) * times.upstream_time =
  (journey.boat_speed + journey.current_speed) * times.downstream_time

/-- The main theorem to prove --/
theorem boat_speed_is_twelve (times : JourneyTimes) 
    (h1 : times.upstream_time = 5)
    (h2 : times.downstream_time = 3) :
    ∃ (journey : RiverJourney), 
      journey.boat_speed = 12 ∧ 
      is_consistent_speed journey times := by
  sorry

end boat_speed_is_twelve_l269_26988


namespace radius_of_larger_circle_l269_26997

/-- Two concentric circles with radii r and R, where R = 4r -/
structure ConcentricCircles where
  r : ℝ
  R : ℝ
  h : R = 4 * r

/-- A chord BC tangent to the inner circle -/
structure TangentChord (c : ConcentricCircles) where
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Diameter AC of the larger circle -/
structure Diameter (c : ConcentricCircles) where
  A : ℝ × ℝ
  C : ℝ × ℝ
  h : dist A C = 2 * c.R

theorem radius_of_larger_circle 
  (c : ConcentricCircles) 
  (d : Diameter c) 
  (t : TangentChord c) 
  (h : dist d.A t.B = 8) : 
  c.R = 16 := by
  sorry

end radius_of_larger_circle_l269_26997


namespace smallest_n_for_sum_equation_l269_26987

theorem smallest_n_for_sum_equation : ∃ (n : ℕ), n = 835 ∧ 
  (∀ (S : Finset ℕ), S.card = n → (∀ x ∈ S, x ≥ 1 ∧ x ≤ 999) → 
    (∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      a + 2*b + 3*c = d)) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (T : Finset ℕ), T.card = m ∧ (∀ x ∈ T, x ≥ 1 ∧ x ≤ 999) ∧
      ¬(∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧ 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
        a + 2*b + 3*c = d)) :=
by sorry

end smallest_n_for_sum_equation_l269_26987


namespace ellipse_eccentricity_k_range_l269_26911

theorem ellipse_eccentricity_k_range (k : ℝ) (e : ℝ) :
  (∃ x y : ℝ, x^2 / k + y^2 / 4 = 1) →
  (1/2 < e ∧ e < 1) →
  (0 < k ∧ k < 3) ∨ (16/3 < k) :=
by sorry

end ellipse_eccentricity_k_range_l269_26911


namespace knight_placement_exists_l269_26972

/-- A position on the modified 6x6 board -/
structure Position :=
  (x : Fin 6)
  (y : Fin 6)
  (valid : ¬((x < 2 ∧ y < 2) ∨ (x > 3 ∧ y < 2) ∨ (x < 2 ∧ y > 3) ∨ (x > 3 ∧ y > 3)))

/-- A knight's move -/
def knightMove (p q : Position) : Prop :=
  (abs (p.x - q.x) == 2 ∧ abs (p.y - q.y) == 1) ∨
  (abs (p.x - q.x) == 1 ∧ abs (p.y - q.y) == 2)

/-- A valid knight placement -/
structure KnightPlacement :=
  (positions : Fin 10 → Position × Position)
  (distinct : ∀ i j, i ≠ j → positions i ≠ positions j)
  (canAttack : ∀ i, knightMove (positions i).1 (positions i).2)

/-- The main theorem -/
theorem knight_placement_exists : ∃ (k : KnightPlacement), True :=
sorry

end knight_placement_exists_l269_26972


namespace geometric_sequence_middle_term_l269_26922

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_middle_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 2)
  (h_a8 : a 8 = 32) :
  a 5 = 8 := by
sorry

end geometric_sequence_middle_term_l269_26922


namespace stating_race_outcomes_count_l269_26912

/-- Represents the number of participants in the race -/
def total_participants : ℕ := 6

/-- Represents the number of top positions we're considering -/
def top_positions : ℕ := 3

/-- Represents the number of participants eligible for top positions -/
def eligible_participants : ℕ := total_participants - 1

/-- 
Calculates the number of different outcomes for top positions in a race
given the number of eligible participants and the number of top positions,
assuming no ties.
-/
def race_outcomes (eligible : ℕ) (positions : ℕ) : ℕ :=
  (eligible - positions + 1).factorial / (eligible - positions).factorial

/-- 
Theorem stating that the number of different 1st-2nd-3rd place outcomes
in a race with 6 participants, where one participant cannot finish 
in the top three and there are no ties, is equal to 60.
-/
theorem race_outcomes_count : 
  race_outcomes eligible_participants top_positions = 60 := by
  sorry

end stating_race_outcomes_count_l269_26912


namespace quadratic_equations_solution_l269_26985

theorem quadratic_equations_solution :
  -- Part 1
  (∀ x, 1969 * x^2 - 1974 * x + 5 = 0 ↔ x = 1 ∨ x = 5/1969) ∧
  -- Part 2
  (∀ a b c x,
    -- Case 1
    (a + b - 2*c = 0 ∧ b + c - 2*a ≠ 0 →
      (a + b - 2*c) * x^2 + (b + c - 2*a) * x + (c + a - 2*b) = 0 ↔
      x = -(c + a - 2*b) / (b + c - 2*a)) ∧
    (a + b - 2*c = 0 ∧ b + c - 2*a = 0 →
      (a + b - 2*c) * x^2 + (b + c - 2*a) * x + (c + a - 2*b) = 0 ↔
      True) ∧
    -- Case 2
    (a + b - 2*c ≠ 0 →
      (a + b - 2*c) * x^2 + (b + c - 2*a) * x + (c + a - 2*b) = 0 ↔
      x = 1 ∨ x = (c + a - 2*b) / (a + b - 2*c))) :=
by sorry

end quadratic_equations_solution_l269_26985


namespace hyperbola_equation_l269_26939

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_focal : 4 * Real.sqrt 5 = 2 * Real.sqrt ((a^2 + b^2) : ℝ))
  (h_asymptote : b / a = 2) :
  a^2 = 4 ∧ b^2 = 16 := by
sorry

end hyperbola_equation_l269_26939


namespace circle_triangle_areas_l269_26907

theorem circle_triangle_areas (a b c : ℝ) (A B C : ℝ) : 
  a = 15 → b = 20 → c = 25 →
  a^2 + b^2 = c^2 →
  A > 0 → B > 0 → C > 0 →
  C > A ∧ C > B →
  A + B + (1/2 * a * b) = C := by
  sorry

end circle_triangle_areas_l269_26907


namespace trig_identity_l269_26909

theorem trig_identity (α : Real) (h : 2 * Real.sin α + Real.cos α = 0) :
  2 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 5 * Real.cos α ^ 2 = -12/5 := by
  sorry

end trig_identity_l269_26909


namespace equation_system_solution_l269_26936

theorem equation_system_solution : ∃! (a b c d e f : ℕ),
  (a ∈ Finset.range 10) ∧
  (b ∈ Finset.range 10) ∧
  (c ∈ Finset.range 10) ∧
  (d ∈ Finset.range 10) ∧
  (e ∈ Finset.range 10) ∧
  (f ∈ Finset.range 10) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧
  (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧
  (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧
  (d ≠ e) ∧ (d ≠ f) ∧
  (e ≠ f) ∧
  (20 * (a - 8) = 20) ∧
  (b / 2 + 17 = 20) ∧
  (c * 8 - 4 = 20) ∧
  ((d + 8) / 12 = 1) ∧
  (4 * e = 20) ∧
  (20 * (f - 2) = 100) :=
by
  sorry


end equation_system_solution_l269_26936


namespace editing_posting_time_is_zero_l269_26929

/-- Represents the time in hours for various activities in video production -/
structure VideoProductionTime where
  setup : ℝ
  painting : ℝ
  cleanup : ℝ
  total : ℝ

/-- The time spent on editing and posting each video -/
def editingPostingTime (t : VideoProductionTime) : ℝ :=
  t.total - (t.setup + t.painting + t.cleanup)

/-- Theorem stating that the editing and posting time is 0 hours -/
theorem editing_posting_time_is_zero (t : VideoProductionTime)
  (h_setup : t.setup = 1)
  (h_painting : t.painting = 1)
  (h_cleanup : t.cleanup = 1)
  (h_total : t.total = 3) :
  editingPostingTime t = 0 := by
  sorry

end editing_posting_time_is_zero_l269_26929


namespace same_last_digit_l269_26927

theorem same_last_digit (a b : ℕ) : 
  (2 * a + b) % 10 = (2 * b + a) % 10 → a % 10 = b % 10 := by
  sorry

end same_last_digit_l269_26927


namespace equation_solution_l269_26940

/-- Given an equation y = a + b / x^2, where a and b are constants,
    if y = 2 when x = -2 and y = 4 when x = -4, then a + b = -6 -/
theorem equation_solution (a b : ℝ) : 
  (2 = a + b / (-2)^2) → 
  (4 = a + b / (-4)^2) → 
  a + b = -6 := by
  sorry

end equation_solution_l269_26940


namespace x_squared_plus_reciprocal_l269_26991

theorem x_squared_plus_reciprocal (x : ℝ) (h : 49 = x^6 + 1/x^6) :
  x^2 + 1/x^2 = (51 : ℝ)^(1/3) := by
  sorry

end x_squared_plus_reciprocal_l269_26991


namespace unique_b_solution_l269_26900

theorem unique_b_solution (a b : ℕ) : 
  0 ≤ a → a < 2^2008 → 0 ≤ b → b < 8 → 
  (7 * (a + 2^2008 * b)) % 2^2011 = 1 → 
  b = 3 := by
sorry

end unique_b_solution_l269_26900


namespace age_of_replaced_man_is_44_l269_26983

/-- The age of the other replaced man given the conditions of the problem -/
def age_of_replaced_man (initial_men_count : ℕ) (age_increase : ℕ) (known_man_age : ℕ) (women_avg_age : ℕ) : ℕ :=
  44

/-- Theorem stating that the age of the other replaced man is 44 years old -/
theorem age_of_replaced_man_is_44 
  (initial_men_count : ℕ) 
  (age_increase : ℕ) 
  (known_man_age : ℕ) 
  (women_avg_age : ℕ) 
  (h1 : initial_men_count = 6)
  (h2 : age_increase = 3)
  (h3 : known_man_age = 24)
  (h4 : women_avg_age = 34) :
  age_of_replaced_man initial_men_count age_increase known_man_age women_avg_age = 44 := by
  sorry

end age_of_replaced_man_is_44_l269_26983


namespace mark_fish_problem_l269_26949

/-- Given the number of tanks, pregnant fish per tank, and young per fish, 
    calculate the total number of young fish. -/
def total_young_fish (num_tanks : ℕ) (fish_per_tank : ℕ) (young_per_fish : ℕ) : ℕ :=
  num_tanks * fish_per_tank * young_per_fish

/-- Theorem stating that with 3 tanks, 4 pregnant fish per tank, and 20 young per fish, 
    the total number of young fish is 240. -/
theorem mark_fish_problem : 
  total_young_fish 3 4 20 = 240 := by
  sorry

end mark_fish_problem_l269_26949


namespace star_calculation_l269_26961

-- Define the ⋆ operation
def star (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem star_calculation : star (star 2 1) 4 = 259 := by
  sorry

end star_calculation_l269_26961


namespace tan_equation_solution_l269_26959

theorem tan_equation_solution (θ : Real) (h1 : 0 < θ) (h2 : θ < Real.pi / 6)
  (h3 : Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) = 0) :
  Real.tan θ = 1 / Real.sqrt 3 := by
sorry

end tan_equation_solution_l269_26959


namespace inscribed_square_area_l269_26904

/-- Given an isosceles right triangle with a square inscribed as described in Figure 1
    with an area of 256 cm², prove that the area of the square inscribed as described
    in Figure 2 is 576 - 256√2 cm². -/
theorem inscribed_square_area (s : ℝ) (h1 : s^2 = 256) : ∃ S : ℝ,
  S^2 = 576 - 256 * Real.sqrt 2 :=
by sorry

end inscribed_square_area_l269_26904


namespace decorative_gravel_cost_l269_26967

/-- The cost of decorative gravel in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of gravel -/
def cubic_yards : ℝ := 8

/-- The total cost of the decorative gravel -/
def total_cost : ℝ := cubic_yards * cubic_feet_per_cubic_yard * cost_per_cubic_foot

theorem decorative_gravel_cost : total_cost = 1728 := by
  sorry

end decorative_gravel_cost_l269_26967


namespace intersection_distance_l269_26906

theorem intersection_distance : ∃ (p₁ p₂ : ℝ × ℝ),
  (p₁.1^2 + p₁.2 = 10 ∧ p₁.1 + p₁.2 = 10) ∧
  (p₂.1^2 + p₂.2 = 10 ∧ p₂.1 + p₂.2 = 10) ∧
  p₁ ≠ p₂ ∧
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = Real.sqrt 2 :=
by sorry

end intersection_distance_l269_26906


namespace oil_tank_capacity_oil_tank_capacity_proof_l269_26924

theorem oil_tank_capacity : ℝ → Prop :=
  fun t => 
    (∃ o : ℝ, o / t = 1 / 6 ∧ (o + 4) / t = 1 / 3) → t = 24

-- The proof is omitted
theorem oil_tank_capacity_proof : oil_tank_capacity 24 :=
  sorry

end oil_tank_capacity_oil_tank_capacity_proof_l269_26924


namespace infinite_solutions_imply_values_l269_26964

theorem infinite_solutions_imply_values (a b : ℚ) : 
  (∀ x : ℚ, a * (2 * x + b) = 12 * x + 5) → 
  (a = 6 ∧ b = 5/6) := by
sorry

end infinite_solutions_imply_values_l269_26964


namespace simplify_and_evaluate_l269_26958

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 - 2) :
  1 - (a - 2) / a / ((a^2 - 4) / (a^2 + a)) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l269_26958


namespace sqrt_three_solution_l269_26901

theorem sqrt_three_solution (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 3*a) : a = Real.sqrt 3 := by
  sorry

end sqrt_three_solution_l269_26901


namespace bug_path_theorem_l269_26952

/-- Represents a rectangular garden paved with square pavers -/
structure PavedGarden where
  width : ℕ  -- width in feet
  length : ℕ  -- length in feet
  paver_size : ℕ  -- size of square paver in feet

/-- Calculates the number of pavers a bug visits when walking diagonally across the garden -/
def pavers_visited (garden : PavedGarden) : ℕ :=
  let width_pavers := garden.width / garden.paver_size
  let length_pavers := (garden.length + garden.paver_size - 1) / garden.paver_size
  width_pavers + length_pavers - Nat.gcd width_pavers length_pavers

/-- Theorem stating that a bug walking diagonally across a 14x19 garden with 2-foot pavers visits 16 pavers -/
theorem bug_path_theorem :
  let garden : PavedGarden := { width := 14, length := 19, paver_size := 2 }
  pavers_visited garden = 16 := by sorry

end bug_path_theorem_l269_26952


namespace some_number_added_l269_26947

theorem some_number_added (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ + a)^2 / (3 * x₁ + 65) = 2 ∧ 
                (x₂ + a)^2 / (3 * x₂ + 65) = 2 ∧ 
                |x₁ - x₂| = 22) → 
  a = 3 := by sorry

end some_number_added_l269_26947


namespace first_course_cost_proof_l269_26902

/-- The cost of Amelia's dinner --/
def dinner_cost : ℝ := 60

/-- The amount Amelia has left after buying all meals --/
def remaining_amount : ℝ := 20

/-- The additional cost of the second course compared to the first --/
def second_course_additional_cost : ℝ := 5

/-- The ratio of the dessert cost to the second course cost --/
def dessert_ratio : ℝ := 0.25

/-- The cost of the first course --/
def first_course_cost : ℝ := 15

theorem first_course_cost_proof :
  ∃ (x : ℝ),
    x = first_course_cost ∧
    dinner_cost - remaining_amount = x + (x + second_course_additional_cost) + dessert_ratio * (x + second_course_additional_cost) :=
by sorry

end first_course_cost_proof_l269_26902


namespace intersection_of_A_and_B_l269_26992

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l269_26992


namespace library_books_remaining_l269_26916

/-- The number of books remaining in a library after a series of events --/
def remaining_books (initial : ℕ) (taken_out : ℕ) (returned : ℕ) (withdrawn : ℕ) : ℕ :=
  initial - taken_out + returned - withdrawn

/-- Theorem stating that given the specific events, 150 books remain in the library --/
theorem library_books_remaining : remaining_books 250 120 35 15 = 150 := by
  sorry

end library_books_remaining_l269_26916


namespace system_of_equations_solution_l269_26910

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = 3) ∧ (7 * x - 3 * y = 20) ∧ x = 11 ∧ y = 19 := by
  sorry

end system_of_equations_solution_l269_26910


namespace unique_right_triangle_completion_l269_26935

/-- A function that checks if three side lengths form a right triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that there is exactly one integer side length 
    that can complete a right triangle with sides 8 and 15 -/
theorem unique_right_triangle_completion :
  ∃! x : ℕ, is_right_triangle 8 15 x :=
sorry

end unique_right_triangle_completion_l269_26935


namespace inequality_solution_set_l269_26966

theorem inequality_solution_set (x : ℝ) : 
  (x + 2) / (x - 1) ≤ 0 ↔ x ∈ Set.Icc (-2) 1 ∧ x ≠ 1 :=
by sorry

end inequality_solution_set_l269_26966


namespace sine_shift_left_l269_26903

/-- Shifting a sine function to the left -/
theorem sine_shift_left (x : ℝ) :
  let f (t : ℝ) := Real.sin t
  let shift : ℝ := π / 6
  let g (t : ℝ) := f (t + shift)
  g x = Real.sin (x + π / 6) :=
by sorry

end sine_shift_left_l269_26903


namespace bracelets_count_l269_26944

def total_stones : ℕ := 140
def stones_per_bracelet : ℕ := 14

theorem bracelets_count : total_stones / stones_per_bracelet = 10 := by
  sorry

end bracelets_count_l269_26944


namespace min_value_of_expression_l269_26982

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  1 / (x + 1) + 4 / (y + 2) ≥ 9 / 4 := by
  sorry

end min_value_of_expression_l269_26982


namespace quadratic_intercept_distance_l269_26971

/-- Given a quadratic function f(x) = x² + ax + b, where the line from (0, b) to one x-intercept
    is perpendicular to y = x, prove that the distance from (0, 0) to the other x-intercept is 1. -/
theorem quadratic_intercept_distance (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  let x₁ := -b  -- One x-intercept
  let x₂ := 1   -- The other x-intercept (to be proven)
  (∀ x, f x = 0 → x = x₁ ∨ x = x₂) →  -- x₁ and x₂ are the only roots
  (x₁ + x₂ = -a ∧ x₁ * x₂ = b) →      -- Vieta's formulas
  (b ≠ 0) →                           -- Ensuring non-zero y-intercept
  (∀ x y, y = -x + b → f x = y) →     -- Line from (0, b) to (x₁, 0) has equation y = -x + b
  x₂ = 1 :=
by sorry

end quadratic_intercept_distance_l269_26971


namespace perpendicular_tangent_line_l269_26930

/-- Given a line L1 with equation x + 3y - 10 = 0 and a circle C with equation x^2 + y^2 = 4,
    prove that a line L2 perpendicular to L1 and tangent to C has the equation 3x - y ± 2√10 = 0 -/
theorem perpendicular_tangent_line 
  (L1 : ℝ → ℝ → Prop) 
  (C : ℝ → ℝ → Prop)
  (h1 : ∀ x y, L1 x y ↔ x + 3*y - 10 = 0)
  (h2 : ∀ x y, C x y ↔ x^2 + y^2 = 4) :
  ∃ L2 : ℝ → ℝ → Prop,
    (∀ x y, L2 x y ↔ (3*x - y = 2*Real.sqrt 10 ∨ 3*x - y = -2*Real.sqrt 10)) ∧
    (∀ x y, L1 x y → ∀ u v, L2 u v → (x - u) * (3 * (y - v)) = -(y - v) * (x - u)) ∧
    (∃ p q, L2 p q ∧ C p q ∧ ∀ x y, C x y → (x - p)^2 + (y - q)^2 ≥ 0) :=
by
  sorry

end perpendicular_tangent_line_l269_26930


namespace insect_count_proof_l269_26979

/-- Calculates the number of insects given the total number of legs and legs per insect -/
def number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Proves that given 48 insect legs and 6 legs per insect, the number of insects is 8 -/
theorem insect_count_proof :
  let total_legs : ℕ := 48
  let legs_per_insect : ℕ := 6
  number_of_insects total_legs legs_per_insect = 8 := by
  sorry

end insect_count_proof_l269_26979


namespace square_land_area_l269_26954

/-- Given a square land with perimeter p and area A, prove that A = 81 --/
theorem square_land_area (p A : ℝ) : p = 36 ∧ 5 * A = 10 * p + 45 → A = 81 := by
  sorry

end square_land_area_l269_26954


namespace board_pair_positive_l269_26969

inductive BoardPair : ℚ × ℚ → Prop where
  | initial : BoardPair (1, 1)
  | trans1a (x y : ℚ) : BoardPair (x, y - 1) → BoardPair (x + y, y + 1)
  | trans1b (x y : ℚ) : BoardPair (x + y, y + 1) → BoardPair (x, y - 1)
  | trans2a (x y : ℚ) : BoardPair (x, x * y) → BoardPair (1 / x, y)
  | trans2b (x y : ℚ) : BoardPair (1 / x, y) → BoardPair (x, x * y)

theorem board_pair_positive (a b : ℚ) : BoardPair (a, b) → a > 0 := by
  sorry

end board_pair_positive_l269_26969


namespace rectangular_garden_dimensions_l269_26919

theorem rectangular_garden_dimensions (perimeter area fixed_side : ℝ) :
  perimeter = 60 →
  area = 200 →
  fixed_side = 10 →
  ∃ (adjacent_side : ℝ),
    adjacent_side = 20 ∧
    2 * (fixed_side + adjacent_side) = perimeter ∧
    fixed_side * adjacent_side = area :=
by sorry

end rectangular_garden_dimensions_l269_26919


namespace jack_payment_l269_26962

/-- The amount Jack paid for sandwiches -/
def amount_paid : ℕ := sorry

/-- The number of sandwiches Jack ordered -/
def num_sandwiches : ℕ := 3

/-- The cost of each sandwich in dollars -/
def cost_per_sandwich : ℕ := 5

/-- The amount of change Jack received in dollars -/
def change_received : ℕ := 5

/-- Theorem stating that the amount Jack paid is $20 -/
theorem jack_payment : amount_paid = 20 := by
  sorry

end jack_payment_l269_26962


namespace right_triangle_among_given_sets_l269_26990

-- Define a function to check if three numbers can form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem statement
theorem right_triangle_among_given_sets :
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 6 7) ∧
  ¬(is_right_triangle 5 (-11) 12) ∧
  is_right_triangle 5 12 13 :=
by sorry

end right_triangle_among_given_sets_l269_26990


namespace sum_of_squares_divided_by_365_l269_26951

theorem sum_of_squares_divided_by_365 : (10^2 + 11^2 + 12^2 + 13^2 + 14^2) / 365 = 2 := by
  sorry

end sum_of_squares_divided_by_365_l269_26951


namespace keith_bought_22_cards_l269_26932

/-- The number of baseball cards Keith bought -/
def cards_bought (initial_cards remaining_cards : ℕ) : ℕ :=
  initial_cards - remaining_cards

/-- Theorem stating that Keith bought 22 baseball cards -/
theorem keith_bought_22_cards : cards_bought 40 18 = 22 := by
  sorry

end keith_bought_22_cards_l269_26932


namespace shipping_cost_per_unit_l269_26921

/-- Proves that the shipping cost per unit is $1.67 given the manufacturing conditions --/
theorem shipping_cost_per_unit 
  (production_cost : ℝ) 
  (fixed_cost : ℝ) 
  (units_sold : ℝ) 
  (selling_price : ℝ) 
  (h1 : production_cost = 80)
  (h2 : fixed_cost = 16500)
  (h3 : units_sold = 150)
  (h4 : selling_price = 191.67)
  : ∃ (shipping_cost : ℝ), 
    shipping_cost = 1.67 ∧ 
    units_sold * (production_cost + shipping_cost) + fixed_cost ≤ units_sold * selling_price ∧
    ∀ (s : ℝ), s < shipping_cost → 
      units_sold * (production_cost + s) + fixed_cost < units_sold * selling_price :=
by sorry

end shipping_cost_per_unit_l269_26921


namespace ellipse_equation_l269_26963

/-- Given an ellipse with center at the origin, foci on the x-axis, 
    major axis length of 4, and minor axis length of 2, 
    its equation is x²/4 + y² = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let center := (0 : ℝ × ℝ)
  let major_axis := 4
  let minor_axis := 2
  let foci_on_x_axis := true
  x^2 / 4 + y^2 = 1 :=
by sorry

end ellipse_equation_l269_26963


namespace quadratic_equation_roots_range_l269_26942

/-- The range of k for which the quadratic equation (k+1)x^2 - 2x + 1 = 0 has two real roots -/
theorem quadratic_equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k + 1) * x₁^2 - 2 * x₁ + 1 = 0 ∧ 
    (k + 1) * x₂^2 - 2 * x₂ + 1 = 0) ↔ 
  (k ≤ 0 ∧ k ≠ -1) :=
sorry

end quadratic_equation_roots_range_l269_26942


namespace impossible_inequality_l269_26970

theorem impossible_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_log : Real.log x / Real.log 2 = Real.log y / Real.log 3 ∧ 
           Real.log y / Real.log 3 = Real.log z / Real.log 5 ∧
           Real.log z / Real.log 5 > 0) :
  ¬(y / 3 < z / 5 ∧ z / 5 < x / 2) := by
sorry

end impossible_inequality_l269_26970


namespace factorization_a_squared_plus_2a_l269_26905

theorem factorization_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a*(a+2) := by
  sorry

end factorization_a_squared_plus_2a_l269_26905


namespace fresh_driving_hours_l269_26925

/-- Calculates the number of hours driving fresh given total distance, total time, and speeds -/
theorem fresh_driving_hours (total_distance : ℝ) (total_time : ℝ) (fresh_speed : ℝ) (fatigued_speed : ℝ) 
  (h1 : total_distance = 152)
  (h2 : total_time = 9)
  (h3 : fresh_speed = 25)
  (h4 : fatigued_speed = 15) :
  ∃ x : ℝ, x = 17 / 10 ∧ fresh_speed * x + fatigued_speed * (total_time - x) = total_distance :=
by
  sorry

end fresh_driving_hours_l269_26925


namespace cubic_inequality_solution_l269_26977

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 9*x^2 + 23*x - 15 < 0 ↔ x ∈ Set.Iio 1 ∪ Set.Ioo 3 5 :=
sorry

end cubic_inequality_solution_l269_26977


namespace greatest_sum_consecutive_integers_l269_26920

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 1000) → (∀ m : ℕ, m * (m + 1) < 1000 → n + (n + 1) ≥ m + (m + 1)) → 
  n + (n + 1) = 63 := by
  sorry

end greatest_sum_consecutive_integers_l269_26920


namespace olivia_weekly_earnings_l269_26914

/-- Olivia's weekly earnings calculation -/
theorem olivia_weekly_earnings 
  (hourly_wage : ℕ) 
  (monday_hours wednesday_hours friday_hours : ℕ) : 
  hourly_wage = 9 → 
  monday_hours = 4 → 
  wednesday_hours = 3 → 
  friday_hours = 6 → 
  hourly_wage * (monday_hours + wednesday_hours + friday_hours) = 117 := by
  sorry

end olivia_weekly_earnings_l269_26914


namespace smallest_number_with_remainders_l269_26975

theorem smallest_number_with_remainders : ∃! N : ℕ,
  (N > 0) ∧
  (N % 13 = 2) ∧
  (N % 15 = 4) ∧
  (N % 17 = 6) ∧
  (N % 19 = 8) ∧
  (∀ M : ℕ, M > 0 ∧ M % 13 = 2 ∧ M % 15 = 4 ∧ M % 17 = 6 ∧ M % 19 = 8 → M ≥ N) ∧
  N = 1070747 :=
by sorry

end smallest_number_with_remainders_l269_26975


namespace count_divisors_5940_mult_6_l269_26998

/-- The number of positive divisors of 5940 that are multiples of 6 -/
def divisors_5940_mult_6 : ℕ := 24

/-- 5940 expressed as a product of prime factors -/
def factorization_5940 : ℕ := 2^2 * 3^3 * 5 * 11

theorem count_divisors_5940_mult_6 :
  (∀ d : ℕ, d > 0 ∧ d ∣ factorization_5940 ∧ 6 ∣ d) →
  (∃! n : ℕ, n = divisors_5940_mult_6) :=
sorry

end count_divisors_5940_mult_6_l269_26998


namespace pure_imaginary_fraction_implies_a_eq_two_l269_26955

/-- If (1 + ai) / (2 - i) is a pure imaginary number, then a = 2 -/
theorem pure_imaginary_fraction_implies_a_eq_two (a : ℝ) :
  (∃ b : ℝ, (1 + a * Complex.I) / (2 - Complex.I) = b * Complex.I) →
  a = 2 := by
sorry

end pure_imaginary_fraction_implies_a_eq_two_l269_26955


namespace no_integer_solutions_l269_26931

theorem no_integer_solutions (c : ℕ) (hc_pos : c > 0) (hc_odd : Odd c) :
  ¬∃ (x y : ℤ), x^2 - y^3 = (2*c)^3 - 1 :=
by sorry

end no_integer_solutions_l269_26931


namespace units_digit_of_product_l269_26917

theorem units_digit_of_product (n m : ℕ) : (5^7 * 6^4) % 10 = 0 := by
  sorry

end units_digit_of_product_l269_26917


namespace laptop_price_theorem_l269_26913

/-- The sticker price of the laptop. -/
def sticker_price : ℝ := 250

/-- The price at store A after discount and rebate. -/
def price_A (x : ℝ) : ℝ := 0.8 * x - 100

/-- The price at store B after discount and rebate. -/
def price_B (x : ℝ) : ℝ := 0.7 * x - 50

/-- Theorem stating that the sticker price satisfies the given conditions. -/
theorem laptop_price_theorem : 
  price_A sticker_price = price_B sticker_price - 25 := by
  sorry

#check laptop_price_theorem

end laptop_price_theorem_l269_26913


namespace square_side_length_l269_26943

theorem square_side_length (perimeter area : ℝ) (h_perimeter : perimeter = 48) (h_area : area = 144) :
  ∃ (side : ℝ), side * 4 = perimeter ∧ side * side = area ∧ side = 12 := by
  sorry

end square_side_length_l269_26943


namespace time_to_cover_distance_l269_26965

/-- Given a constant rate of movement and a remaining distance, prove that the time to cover the remaining distance can be calculated by dividing the remaining distance by the rate. -/
theorem time_to_cover_distance (rate : ℝ) (distance : ℝ) (time : ℝ) : 
  rate > 0 → distance > 0 → time = distance / rate → time * rate = distance := by sorry

end time_to_cover_distance_l269_26965


namespace equation_satisfied_for_all_x_l269_26956

theorem equation_satisfied_for_all_x (a b c x : ℝ) 
  (h : a / b = 2 ∧ b / c = 3/4) : 
  (a + b) * (c - x) / a^2 - (b + c) * (x - 2*c) / (b*c) - 
  (c + a) * (c - 2*x) / (a*c) = (a + b) * c / (a*b) + 2 := by
  sorry

end equation_satisfied_for_all_x_l269_26956


namespace perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l269_26945

/-- Two lines in the plane -/
structure Lines (m : ℝ) where
  l1 : ℝ → ℝ → Prop
  l2 : ℝ → ℝ → Prop
  eq1 : ∀ x y, l1 x y ↔ x + m * y + 6 = 0
  eq2 : ∀ x y, l2 x y ↔ (m - 2) * x + 3 * y + 2 * m = 0

/-- The lines are perpendicular -/
def Perpendicular (m : ℝ) (lines : Lines m) : Prop :=
  (-1 / m) * ((m - 2) / 3) = -1

/-- The lines are parallel -/
def Parallel (m : ℝ) (lines : Lines m) : Prop :=
  -1 / m = (m - 2) / 3

theorem perpendicular_implies_m_eq_half (m : ℝ) (lines : Lines m) :
  Perpendicular m lines → m = 1 / 2 := by
  sorry

theorem parallel_implies_m_eq_neg_one (m : ℝ) (lines : Lines m) :
  Parallel m lines → m = -1 := by
  sorry

end perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l269_26945


namespace value_of_k_l269_26937

theorem value_of_k (k : ℝ) (h : 16 / k = 4) : k = 4 := by
  sorry

end value_of_k_l269_26937


namespace purely_imaginary_complex_number_l269_26946

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 - 3*m = 0) ∧ (m^2 - 5*m + 6 ≠ 0) → m = 0 := by sorry

end purely_imaginary_complex_number_l269_26946


namespace arithmetic_sequence_sum_l269_26926

/-- Given an arithmetic sequence {a_n} where a₁ + 3a₈ + a₁₅ = 120, prove that a₂ + a₁₄ = 48. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 1 + 3 * a 8 + a 15 = 120 →                      -- given condition
  a 2 + a 14 = 48 := by                             -- conclusion to prove
sorry

end arithmetic_sequence_sum_l269_26926


namespace inconsistent_weight_problem_l269_26960

theorem inconsistent_weight_problem :
  ∀ (initial_students : ℕ) (initial_avg_weight : ℝ) 
    (new_students : ℕ) (new_avg_weight : ℝ) 
    (first_new_student_weight : ℝ) (second_new_student_min_weight : ℝ),
  initial_students = 19 →
  initial_avg_weight = 15 →
  new_students = 2 →
  new_avg_weight = 14.6 →
  first_new_student_weight = 12 →
  second_new_student_min_weight = 14 →
  ¬∃ (second_new_student_weight : ℝ),
    (initial_students * initial_avg_weight + first_new_student_weight + second_new_student_weight) / 
      (initial_students + new_students) = new_avg_weight ∧
    second_new_student_weight ≥ second_new_student_min_weight :=
by sorry

end inconsistent_weight_problem_l269_26960


namespace min_value_expression_l269_26928

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  6 * Real.sqrt (a * b) + 3 / a + 3 / b ≥ 12 := by
  sorry

end min_value_expression_l269_26928


namespace bus_seats_solution_l269_26918

/-- Represents the seating arrangement in a bus -/
structure BusSeats where
  left : ℕ  -- Number of seats on the left side
  right : ℕ  -- Number of seats on the right side
  back : ℕ  -- Capacity of the back seat
  capacity_per_seat : ℕ  -- Number of people each regular seat can hold

/-- The total capacity of the bus -/
def total_capacity (bs : BusSeats) : ℕ :=
  bs.capacity_per_seat * (bs.left + bs.right) + bs.back

theorem bus_seats_solution :
  ∃ (bs : BusSeats),
    bs.right = bs.left - 3 ∧
    bs.capacity_per_seat = 3 ∧
    bs.back = 10 ∧
    total_capacity bs = 91 ∧
    bs.left = 15 := by
  sorry

end bus_seats_solution_l269_26918


namespace line_at_0_l269_26957

/-- A line parameterized by t -/
def line (t : ℝ) : ℝ × ℝ := sorry

/-- The vector on the line at t = 1 is (2, 3) -/
axiom line_at_1 : line 1 = (2, 3)

/-- The vector on the line at t = 4 is (8, -5) -/
axiom line_at_4 : line 4 = (8, -5)

/-- The vector on the line at t = 5 is (10, -9) -/
axiom line_at_5 : line 5 = (10, -9)

/-- The vector on the line at t = 0 is (0, 17/3) -/
theorem line_at_0 : line 0 = (0, 17/3) := by sorry

end line_at_0_l269_26957


namespace circle_equation_with_given_diameter_l269_26968

/-- The standard equation of a circle with diameter endpoints A(-1, 2) and B(5, -6) -/
theorem circle_equation_with_given_diameter :
  ∃ (f : ℝ × ℝ → ℝ),
    (∀ x y : ℝ, f (x, y) = (x - 2)^2 + (y + 2)^2) ∧
    (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | f p = 25} ↔ 
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
        x = -1 + 6*t ∧ 
        y = 2 - 8*t) := by
  sorry

end circle_equation_with_given_diameter_l269_26968


namespace circle_angle_problem_l269_26934

theorem circle_angle_problem (x y : ℝ) : 
  3 * x + 2 * y + 5 * x + 7 * x = 360 →
  x = y →
  x = 360 / 17 ∧ y = 360 / 17 := by
sorry

end circle_angle_problem_l269_26934


namespace solar_systems_per_planet_l269_26938

theorem solar_systems_per_planet (total_bodies : ℕ) (planets : ℕ) : 
  total_bodies = 200 → planets = 20 → (total_bodies - planets) / planets = 9 := by
sorry

end solar_systems_per_planet_l269_26938


namespace angle_between_given_lines_l269_26981

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 3 * y + 3 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the angle between two lines
def angle_between_lines (l1 l2 : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem angle_between_given_lines :
  angle_between_lines line1 line2 = Real.arctan (1 / 2) := by sorry

end angle_between_given_lines_l269_26981


namespace debbys_friend_photos_l269_26923

theorem debbys_friend_photos (total_photos family_photos : ℕ) 
  (h1 : total_photos = 86) 
  (h2 : family_photos = 23) : 
  total_photos - family_photos = 63 := by
  sorry

end debbys_friend_photos_l269_26923


namespace intersection_implies_a_value_l269_26908

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 := by
sorry

end intersection_implies_a_value_l269_26908


namespace discount_difference_l269_26974

def original_bill : ℝ := 12000

def single_discount (bill : ℝ) : ℝ := bill * 0.7

def successive_discounts (bill : ℝ) : ℝ := bill * 0.75 * 0.95

theorem discount_difference :
  successive_discounts original_bill - single_discount original_bill = 150 := by
  sorry

end discount_difference_l269_26974


namespace third_side_is_fifteen_l269_26941

/-- A triangle with two known sides and perimeter -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  perimeter : ℝ

/-- Calculate the third side of a triangle given two sides and the perimeter -/
def thirdSide (t : Triangle) : ℝ :=
  t.perimeter - t.side1 - t.side2

/-- Theorem: The third side of the specific triangle is 15 -/
theorem third_side_is_fifteen : 
  let t : Triangle := { side1 := 7, side2 := 10, perimeter := 32 }
  thirdSide t = 15 := by
  sorry

end third_side_is_fifteen_l269_26941


namespace june_bike_ride_l269_26978

/-- Given that June rides her bike at a constant rate and travels 2 miles in 6 minutes,
    prove that she will travel 5 miles in 15 minutes. -/
theorem june_bike_ride (rate : ℚ) : 
  (2 : ℚ) / (6 : ℚ) = rate → (5 : ℚ) / rate = (15 : ℚ) := by
  sorry

end june_bike_ride_l269_26978


namespace three_integers_ratio_l269_26984

theorem three_integers_ratio : ∀ (a b c : ℤ),
  (a : ℚ) / b = 2 / 5 ∧ 
  (b : ℚ) / c = 5 / 8 ∧ 
  ((a + 6 : ℚ) / b = 1 / 3) →
  a = 36 ∧ b = 90 ∧ c = 144 :=
by sorry

end three_integers_ratio_l269_26984


namespace pure_imaginary_ratio_l269_26996

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 4*I) * (a + b*I) = y*I) : a/b = -4/3 := by
  sorry

end pure_imaginary_ratio_l269_26996


namespace modified_cube_edges_l269_26995

/-- Represents a cube with a given side length. -/
structure Cube where
  sideLength : ℕ

/-- Represents the structure after removing unit cubes from corners. -/
structure ModifiedCube where
  originalCube : Cube
  removedCubeSize : ℕ

/-- Calculates the number of edges in the modified cube structure. -/
def edgesInModifiedCube (mc : ModifiedCube) : ℕ :=
  12 * 3  -- Each original edge is divided into 3 segments

/-- Theorem stating that a cube of side length 4 with unit cubes removed from corners has 36 edges. -/
theorem modified_cube_edges :
  ∀ (mc : ModifiedCube),
    mc.originalCube.sideLength = 4 →
    mc.removedCubeSize = 1 →
    edgesInModifiedCube mc = 36 := by
  sorry


end modified_cube_edges_l269_26995


namespace f_properties_l269_26980

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + (1 + Real.cos (2 * x)) / 4

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (k : ℤ), ∀ (x : ℝ), 
    k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 →
    ∀ (y : ℝ), k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤ x → f y ≤ f x) ∧
  (∀ (A B C : ℝ) (a b c : ℝ),
    f A = 1/2 → b + c = 3 →
    a = Real.sqrt (b^2 + c^2 - 2*b*c*Real.cos A) →
    a ≥ 3/2) :=
by sorry

end f_properties_l269_26980


namespace only_two_satisfies_condition_l269_26933

def is_quadratic_residue (a p : ℕ) : Prop :=
  ∃ x, x^2 ≡ a [MOD p]

def all_quadratic_residues (p : ℕ) : Prop :=
  ∀ k ∈ Finset.range p, is_quadratic_residue (2 * (p / k) - 1) p

theorem only_two_satisfies_condition :
  ∀ p, Nat.Prime p → (all_quadratic_residues p ↔ p = 2) := by sorry

end only_two_satisfies_condition_l269_26933


namespace lcm_of_8_and_12_l269_26976

theorem lcm_of_8_and_12 :
  let a : ℕ := 8
  let b : ℕ := 12
  let hcf : ℕ := 4
  (Nat.gcd a b = hcf) → (Nat.lcm a b = 24) :=
by
  sorry

end lcm_of_8_and_12_l269_26976


namespace nested_fraction_equality_l269_26986

theorem nested_fraction_equality : 2 - (1 / (2 - (1 / (2 + 2)))) = 10 / 7 := by
  sorry

end nested_fraction_equality_l269_26986


namespace set_intersection_equality_l269_26953

def M : Set ℝ := {x | |x| < 1}
def N : Set ℝ := {x | x^2 - x < 0}

theorem set_intersection_equality : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end set_intersection_equality_l269_26953


namespace sum_of_ratios_geq_six_l269_26989

theorem sum_of_ratios_geq_six {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / y + y / z + z / x + y / x + z / y + x / z ≥ 6 := by
  sorry

end sum_of_ratios_geq_six_l269_26989


namespace inequality_proof_l269_26948

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≥ Real.sqrt (3 / 2) * Real.sqrt (x + y + z) := by
  sorry

end inequality_proof_l269_26948


namespace impossible_cover_all_endings_l269_26999

theorem impossible_cover_all_endings (a : Fin 14 → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) : 
  ¬(∀ d : Fin 100, ∃ k l : Fin 14, (a k + a l) % 100 = d) := by
  sorry

end impossible_cover_all_endings_l269_26999


namespace possible_values_of_a_l269_26950

/-- Given integers a, b, c satisfying the equation (x - a)(x - 5) + 1 = (x + b)(x + c)
    and either (b + 5)(c + 5) = 1 or (b + 5)(c + 5) = 4,
    prove that the possible values of a are 2, 3, 4, and 7. -/
theorem possible_values_of_a (a b c : ℤ) 
  (h1 : ∀ x, (x - a) * (x - 5) + 1 = (x + b) * (x + c))
  (h2 : (b + 5) * (c + 5) = 1 ∨ (b + 5) * (c + 5) = 4) :
  a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 7 := by
  sorry


end possible_values_of_a_l269_26950


namespace copy_pages_theorem_l269_26994

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 3

/-- The budget in dollars -/
def budget : ℕ := 15

/-- The maximum number of pages that can be copied -/
def max_pages : ℕ := budget * 100 / cost_per_page

theorem copy_pages_theorem : max_pages = 500 := by
  sorry

end copy_pages_theorem_l269_26994


namespace two_parts_problem_l269_26915

theorem two_parts_problem (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : x = 13 := by
  sorry

end two_parts_problem_l269_26915


namespace people_in_room_l269_26973

theorem people_in_room (empty_chairs : ℕ) 
  (h1 : empty_chairs = 5)
  (h2 : ∃ (total_chairs : ℕ), empty_chairs = total_chairs / 5)
  (h3 : ∃ (seated_people : ℕ) (total_people : ℕ), 
    seated_people = 4 * total_chairs / 5 ∧
    seated_people = 5 * total_people / 8) :
  ∃ (total_people : ℕ), total_people = 32 := by
sorry

end people_in_room_l269_26973


namespace prime_square_difference_divisibility_l269_26993

theorem prime_square_difference_divisibility (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧
    1 < p - a^2 ∧
    p - a^2 < p - b^2 ∧
    (p - b^2) % (p - a^2) = 0 := by
  sorry

end prime_square_difference_divisibility_l269_26993
