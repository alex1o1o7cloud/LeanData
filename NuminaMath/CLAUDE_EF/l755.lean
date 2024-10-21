import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l755_75566

/-- The function f(x) = 3cos(x) - √3sin(x) -/
noncomputable def f (x : ℝ) := 3 * Real.cos x - Real.sqrt 3 * Real.sin x

/-- The symmetry equation of the graph of f(x) -/
noncomputable def symmetry_equation := 5 * Real.pi / 6

/-- Theorem stating that the symmetry equation of the graph of f(x) is x = 5π/6 -/
theorem symmetry_of_f : 
  ∀ x : ℝ, f (symmetry_equation + x) = f (symmetry_equation - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l755_75566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_values_l755_75506

-- Define the point A
def A : ℝ × ℝ := (-2, 4)

-- Define the slope of the line (tan(135°) = -1)
def line_slope : ℝ := -1

-- Define the parabola equation
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line equation
def line (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the relationship between vectors
def vector_relation (lambda : ℝ) (M N : ℝ × ℝ) : Prop :=
  (M.1 - A.1, M.2 - A.2) = lambda • (N.1 - M.1, N.2 - M.2) ∧
  (N.1 - A.1, N.2 - A.2) = (1/lambda) • (N.1 - M.1, N.2 - M.2)

theorem intersection_points_values (p lambda : ℝ) (M N : ℝ × ℝ) :
  p > 0 →
  lambda > 0 →
  parabola p M.1 M.2 →
  parabola p N.1 N.2 →
  line M.1 M.2 →
  line N.1 N.2 →
  vector_relation lambda M N →
  p = 1 ∧ lambda = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_values_l755_75506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_game_probabilities_l755_75561

structure LotteryGame where
  boxes : Fin 4 → Bool
  prize_box : Fin 4
  player_choice : Fin 4
  host_opened : Fin 4

noncomputable def probability_host_opens_box_4 (game : LotteryGame) : ℝ :=
  1 / 3

noncomputable def probability_prize_in_box (game : LotteryGame) (box : Fin 4) : ℝ :=
  if box = game.host_opened then 0
  else if box = 1 ∨ box = 3 then 3 / 8
  else 1 / 4

theorem lottery_game_probabilities (game : LotteryGame) 
  (h1 : game.player_choice = 2) 
  (h2 : game.host_opened = 4) :
  probability_host_opens_box_4 game = 1 / 3 ∧
  probability_prize_in_box game 1 > probability_prize_in_box game 2 ∧
  probability_prize_in_box game 3 > probability_prize_in_box game 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_game_probabilities_l755_75561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l755_75586

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 3/2) ^ x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x + 1/(16*a))

def is_decreasing (h : ℝ → ℝ) : Prop := ∀ x y, x < y → h x > h y

def has_domain_reals (h : ℝ → ℝ) : Prop := ∀ x, ∃ y, h x = y

theorem problem_statement (a : ℝ) :
  (¬(is_decreasing (g a) ∧ has_domain_reals (f a))) ∧
  (is_decreasing (g a) ∨ has_domain_reals (f a)) →
  (3/2 < a ∧ a ≤ 2) ∨ (a ≥ 5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l755_75586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_first_group_l755_75562

/-- The work rate of one man -/
noncomputable def man_rate : ℝ := 1

/-- The work rate of one woman -/
noncomputable def woman_rate : ℝ := 1 / 7

/-- The number of women in the first group -/
def x : ℕ := 23

theorem women_in_first_group :
  (3 * man_rate + x * woman_rate = 6 * man_rate + 2 * woman_rate) ∧
  (4 * man_rate + 2 * woman_rate = 5 / 7 * (3 * man_rate + x * woman_rate)) :=
by
  sorry

#eval x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_first_group_l755_75562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_even_implies_m_eq_neg_one_l755_75513

/-- A power function f(x) = (m^2 - m - 1)x^(1-m) where m is a real number -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(1 - m)

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- If f(x) = (m^2 - m - 1)x^(1-m) is an even function, then m = -1 -/
theorem power_function_even_implies_m_eq_neg_one (m : ℝ) :
  is_even_function (f m) → m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_even_implies_m_eq_neg_one_l755_75513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l755_75595

/-- Parabola with chord theorem -/
theorem parabola_chord_theorem (p l : ℝ) (hp : p > 0) :
  let min_distance := λ l' : ℝ => if l' < 2*p then l'^2/(8*p) else (l' - p)/2
  let m_coord := λ l' : ℝ => if l' < 2*p 
                      then (l'^2/(8*p), 0)
                      else ((l' - p)/2, Real.sqrt (p*(l' - 2*p)/2))
  ∀ x y : ℝ, y^2 = 2*p*x →
  ∀ A B : ℝ × ℝ, A.1^2 = 2*p*A.2 ∧ B.1^2 = 2*p*B.2 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = l^2 →
  let M := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  M.1 ≥ min_distance l ∧ 
  (M.1 = min_distance l → M = m_coord l) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_theorem_l755_75595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_distribution_l755_75534

def number_of_photos_given (student : ℕ) : ℕ := sorry

def total_photos_given (class_size : ℕ) : ℕ := sorry

theorem photo_distribution (x : ℕ) : 
  (x > 0) →  -- Ensure there's at least one student
  (x * (x - 1) = 2550) ↔ 
  (∀ (student : ℕ), student < x → (x - 1) = number_of_photos_given student) ∧
  (2550 = total_photos_given x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_distribution_l755_75534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_yards_after_marathons_l755_75555

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon : Marathon := ⟨26, 400⟩

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

/-- Converts a number of marathons to a total distance -/
def marathons_to_distance (n : ℕ) (m : Marathon) : Distance :=
  let total_miles := n * m.miles + (n * m.yards) / yards_per_mile
  let total_yards := (n * m.yards) % yards_per_mile
  ⟨total_miles, total_yards⟩

theorem remaining_yards_after_marathons :
  ∃ (m : ℕ) (y : ℕ), y < yards_per_mile ∧
  Distance.yards (marathons_to_distance num_marathons marathon) = y ∧ y = 720 := by
  sorry

#eval Distance.yards (marathons_to_distance num_marathons marathon)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_yards_after_marathons_l755_75555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_line_divides_green_area_equally_l755_75535

-- Define the rectangles
structure Rectangle where
  center : ℝ × ℝ
  width : ℝ
  height : ℝ

-- Define the problem setup
noncomputable def outer_rectangle : Rectangle := sorry
noncomputable def inner_rectangle : Rectangle := sorry

-- Assumption that inner rectangle is inside outer rectangle
axiom inner_inside_outer : 
  inner_rectangle.center.1 - inner_rectangle.width / 2 ≥ outer_rectangle.center.1 - outer_rectangle.width / 2 ∧
  inner_rectangle.center.1 + inner_rectangle.width / 2 ≤ outer_rectangle.center.1 + outer_rectangle.width / 2 ∧
  inner_rectangle.center.2 - inner_rectangle.height / 2 ≥ outer_rectangle.center.2 - outer_rectangle.height / 2 ∧
  inner_rectangle.center.2 + inner_rectangle.height / 2 ≤ outer_rectangle.center.2 + outer_rectangle.height / 2

-- Define the line passing through the centers
def center_line (r1 r2 : Rectangle) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p = (1 - t) • r1.center + t • r2.center}

-- Define the green area
def green_area : Set (ℝ × ℝ) :=
  {p | p ∈ set_of outer_rectangle ∧ p ∉ set_of inner_rectangle}
  where
    set_of (r : Rectangle) : Set (ℝ × ℝ) :=
      {p | r.center.1 - r.width / 2 ≤ p.1 ∧ p.1 ≤ r.center.1 + r.width / 2 ∧
           r.center.2 - r.height / 2 ≤ p.2 ∧ p.2 ≤ r.center.2 + r.height / 2}

-- Theorem statement
theorem center_line_divides_green_area_equally :
  let l := center_line outer_rectangle inner_rectangle
  ∃ (A₁ A₂ : Set (ℝ × ℝ)), A₁ ∪ A₂ = green_area ∧ A₁ ∩ A₂ = ∅ ∧ 
    (∀ p ∈ green_area, p ∈ A₁ ↔ p ∉ A₂) ∧
    (∀ p ∈ l, p ∈ green_area → p ∈ A₁ ∩ A₂) ∧
    MeasureTheory.volume A₁ = MeasureTheory.volume A₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_line_divides_green_area_equally_l755_75535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_is_sqrt_573_l755_75502

/-- The distance from a point to a plane defined by three points -/
noncomputable def distance_point_to_plane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -A * x₁ - B * y₁ - C * z₁
  Real.sqrt ((A * x₀ + B * y₀ + C * z₀ + D)^2 / (A^2 + B^2 + C^2))

/-- The theorem to be proved -/
theorem distance_to_plane_is_sqrt_573 :
  distance_point_to_plane (3, 6, 68) (-3, -5, 6) (2, 1, -4) (0, -3, -1) = Real.sqrt 573 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_is_sqrt_573_l755_75502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l755_75517

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) / (3 * x - 1)

noncomputable def S (n : ℕ) : ℝ := sorry
noncomputable def T (n : ℕ) : ℝ := sorry

noncomputable def g (n : ℕ) : ℝ := S (2 * n - 1) / T (2 * n - 1)

def b (n : ℕ) : ℝ := 1 + 3 * (n - 1)

theorem problem_solution :
  -- Conditions
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₂ = 3 * x₁ ∧ f (-8) x₁ = -4 * x₁ + 8 ∧ f (-8) x₂ = -4 * x₂ + 8) →
  (∀ n : ℕ, n > 0 → S n / T n = f (-8) n) →
  (S 1 / T 1 = 5/2) →
  -- Conclusions
  (∀ n : ℕ, n > 0 → g n ≤ 5/2) ∧
  (¬ ∃ m k : ℕ, m > 0 ∧ k > 0 ∧ g m = b k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l755_75517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_of_line_l755_75564

theorem polar_equation_of_line (x y ρ θ : ℝ) :
  (((Real.sqrt 3) / 3) * x - y = 0) ∧ (ρ ≥ 0) ∧ (0 ≤ θ) ∧ (θ < 2 * Real.pi) →
  (θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_of_line_l755_75564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jace_speed_l755_75525

/-- Represents Jace's driving scenario -/
structure DrivingScenario where
  first_session : ℚ  -- Duration of the first driving session in hours
  break_duration : ℚ  -- Duration of the break in hours
  second_session : ℚ  -- Duration of the second driving session in hours
  total_distance : ℚ  -- Total distance traveled in miles

/-- Calculates the speed given a driving scenario -/
def calculate_speed (scenario : DrivingScenario) : ℚ :=
  scenario.total_distance / (scenario.first_session + scenario.second_session)

/-- Theorem stating that Jace's speed is 60 miles per hour -/
theorem jace_speed (scenario : DrivingScenario) 
  (h1 : scenario.first_session = 4)
  (h2 : scenario.break_duration = 1/2)
  (h3 : scenario.second_session = 9)
  (h4 : scenario.total_distance = 780) : 
  calculate_speed scenario = 60 := by
  sorry

#check jace_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jace_speed_l755_75525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l755_75501

-- Define the function s(x)
noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^2

-- State the theorem about the range of s(x)
theorem range_of_s :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, x ≠ 2 ∧ s x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_l755_75501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_pipe_fill_time_l755_75512

/-- The time it takes for the first pipe to fill the tank -/
noncomputable def T : ℝ := sorry

/-- The capacity of the tank in liters -/
noncomputable def tank_capacity : ℝ := 675

/-- The rate at which the third pipe drains the tank (in tank portions per minute) -/
noncomputable def drain_rate : ℝ := 45 / tank_capacity

/-- The time it takes to fill the tank when all pipes are open -/
noncomputable def fill_time_all : ℝ := 15

/-- The time it takes for the second pipe to fill the tank -/
noncomputable def fill_time_second : ℝ := 20

theorem first_pipe_fill_time :
  (1 / T) + (1 / fill_time_second) - drain_rate = (1 / fill_time_all) →
  T = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_pipe_fill_time_l755_75512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_theorem_l755_75579

/-- Represents the scale of a map in inches per mile -/
structure MapScale where
  scale : ℚ
  deriving Repr

/-- Calculates the map scale given the map distance, travel time, and average speed -/
def calculateMapScale (mapDistance : ℚ) (travelTime : ℚ) (averageSpeed : ℚ) : MapScale :=
  ⟨mapDistance / (travelTime * averageSpeed)⟩

theorem map_scale_theorem (mapDistance : ℚ) (travelTime : ℚ) (averageSpeed : ℚ)
    (h1 : mapDistance = 5)
    (h2 : travelTime = 7/2)
    (h3 : averageSpeed = 60) :
    (calculateMapScale mapDistance travelTime averageSpeed).scale = 1 / 42 := by
  sorry

#eval calculateMapScale 5 (7/2) 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_theorem_l755_75579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combinations_count_l755_75557

/-- Represents a digit in the range 1 to 6 -/
inductive Digit
| one | two | three | four | five | six

/-- Defines whether a digit is even or odd -/
def Digit.isEven : Digit → Bool
| .two => true
| .four => true
| .six => true
| _ => false

/-- Defines whether a digit is odd -/
def Digit.isOdd (d : Digit) : Bool := !d.isEven

/-- Represents a valid lock combination -/
structure LockCombination where
  digits : Fin 6 → Digit
  alternating : ∀ i : Fin 5, (digits i).isEven ≠ (digits (i.succ)).isEven

/-- Instance to make LockCombination a finite type -/
instance : Fintype LockCombination := by sorry

/-- The main theorem stating the number of possible lock combinations -/
theorem lock_combinations_count :
  Fintype.card LockCombination = 1458 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combinations_count_l755_75557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_for_min_value_l755_75569

open Real

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - log x

-- State the theorem
theorem exists_a_for_min_value :
  ∃ a : ℝ, (∀ x ∈ Set.Ioo 0 (exp 1), f a x ≥ 3/2) ∧
           (∃ x₀ ∈ Set.Ioo 0 (exp 1), f a x₀ = 3/2) ∧
           a = (exp 1)^2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_for_min_value_l755_75569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_l755_75574

/-- The distance between the first and last pole in feet -/
def total_distance : ℚ := 6600

/-- The number of telephone poles -/
def num_poles : ℕ := 51

/-- The number of Elmer's strides between consecutive poles -/
def elmer_strides_per_gap : ℕ := 50

/-- The number of Oscar's leaps between consecutive poles -/
def oscar_leaps_per_gap : ℕ := 15

/-- The length of Elmer's stride in feet -/
noncomputable def elmer_stride_length : ℚ := total_distance / (elmer_strides_per_gap * (num_poles - 1))

/-- The length of Oscar's leap in feet -/
noncomputable def oscar_leap_length : ℚ := total_distance / (oscar_leaps_per_gap * (num_poles - 1))

theorem leap_stride_difference : 
  ⌊(oscar_leap_length - elmer_stride_length : ℚ)⌋ = 6 := by
  sorry

#eval total_distance
#eval num_poles
#eval elmer_strides_per_gap
#eval oscar_leaps_per_gap

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_l755_75574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_intersection_theorem_l755_75527

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A

-- Define the intersection of opposite sides
def opposite_sides_intersect (A B C D E F : ℝ × ℝ) : Prop :=
  (∃ t : ℝ, E = (1 - t) • A + t • B ∧ ∃ s : ℝ, E = (1 - s) • C + s • D) ∧
  (∃ u : ℝ, F = (1 - u) • A + u • D ∧ ∃ v : ℝ, F = (1 - v) • B + v • C)

-- Define the line segment length
noncomputable def segment_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem quadrilateral_intersection_theorem 
  (h1 : is_quadrilateral A B C D)
  (h2 : opposite_sides_intersect A B C D E F) :
  -(segment_length A E * segment_length C E) / (segment_length B E * segment_length D E) = 
   (segment_length A F * segment_length C F) / (segment_length B F * segment_length D F) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_intersection_theorem_l755_75527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_result_l755_75573

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; -4, 5]
def B : Matrix (Fin 2) (Fin 1) ℤ := !![4; -2]
def C : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 0, -1]

theorem matrix_product_result :
  C * (A * B) = !![16; 26] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_result_l755_75573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l755_75592

/-- Calculates the final amount for an investment with compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) (compounds_per_year : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * time)

/-- The initial investment amount -/
def initial_investment : ℝ := 75000

/-- The annual interest rate -/
def annual_rate : ℝ := 0.05

/-- The investment time in years -/
def investment_time : ℝ := 3

/-- Theorem stating the difference between monthly and yearly compounding -/
theorem investment_difference :
  ∃ (diff : ℝ),
    abs (compound_interest initial_investment annual_rate investment_time 12 -
         compound_interest initial_investment annual_rate investment_time 1 - diff) < 0.5 ∧
    diff = 204 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l755_75592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2018_terms_l755_75585

noncomputable def a (n : ℕ) : ℝ :=
  if n % 2 = 1 then 1 / (n^2 + 2*n)
  else Real.sin (n * Real.pi / 4)

noncomputable def S (n : ℕ) : ℝ :=
  (Finset.range n).sum (fun i => a (i + 1))

theorem sum_2018_terms : S 2018 = 3028 / 2019 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2018_terms_l755_75585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l755_75532

/-- The area of a triangle given its three vertices in 2D space -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  (1 / 2) * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (2, -3), (1, 4), and (-3, -2) is 17 -/
theorem triangle_area_example : triangleArea (2, -3) (1, 4) (-3, -2) = 17 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l755_75532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l755_75537

noncomputable section

/-- A geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ :=
  λ n => a * r^(n - 1)

/-- Sum of first n terms of a geometric sequence -/
def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_property (a r : ℝ) :
  let seq := geometric_sequence a r
  let S := geometric_sum a r
  (seq 1 + seq 3 = 5/2) ∧ (seq 2 + seq 4 = 5/4) →
  S 3 / seq 3 = 7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l755_75537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l755_75548

-- Define the sequences a_n and b_n
def a : ℕ → ℤ := sorry
def b : ℕ → ℝ := sorry

-- State the theorem
theorem arithmetic_geometric_sequence :
  (∀ n : ℕ, a (n + 1) < a n) →  -- a_n is decreasing
  (∀ n : ℕ, b n = 2 ^ (a n)) →  -- b_n = 2^(a_n)
  (b 1 * b 2 * b 3 = 64) →      -- b_1 * b_2 * b_3 = 64
  (b 1 + b 2 + b 3 = 14) →      -- b_1 + b_2 + b_3 = 14
  (a 2017 = -2013) :=           -- a_2017 = -2013
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_l755_75548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadratic_polynomial_root_l755_75563

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial := ℤ → ℤ → ℤ → ℝ → ℝ

/-- Definition of a quadratic polynomial -/
def isQuadratic (f : QuadraticPolynomial) : Prop :=
  ∃ a b c : ℤ, ∀ x : ℝ, f a b c x = a * x^2 + b * x + c

/-- The theorem stating the existence of a quadratic polynomial
    with integer coefficients such that f(f(√3)) = 0 -/
theorem exists_quadratic_polynomial_root :
  ∃ f : QuadraticPolynomial, isQuadratic f ∧ f 2 (-8) 0 (f 2 (-8) 0 (Real.sqrt 3)) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadratic_polynomial_root_l755_75563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l755_75571

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The focal length of an ellipse -/
noncomputable def focal_length (e : Ellipse) : ℝ := 
  Real.sqrt (e.a^2 - e.b^2)

/-- Theorem: For an ellipse with equation x²/25 + y²/m² = 1 where m > 0 and focal length 8,
    the value of m is either 3 or √41 -/
theorem ellipse_m_values (m : ℝ) (h_m_pos : m > 0) 
    (h_focal_length : focal_length ⟨5, m, by norm_num, h_m_pos⟩ = 8) :
    m = 3 ∨ m = Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l755_75571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l755_75541

def f (a : ℝ) (x : ℝ) : ℝ := |3*x + 2| - |2*x + a|

theorem problem_statement :
  (∀ x : ℝ, f (4/3) x ≥ 0) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) → a = 4/3) ∧
  (∀ a : ℝ, (∃ x ∈ Set.Icc 1 2, f a x ≤ 0) → 
    (a ≥ 3 ∨ a ≤ -7)) ∧
  (∀ a : ℝ, (a ≥ 3 ∨ a ≤ -7) → 
    (∃ x ∈ Set.Icc 1 2, f a x ≤ 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l755_75541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l755_75514

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -1/8 * x^2

/-- The focus of a parabola -/
noncomputable def focus (h k p : ℝ) : ℝ × ℝ := (h, k + p)

/-- The parameter p for a parabola -/
noncomputable def p_value (a : ℝ) : ℝ := -1 / (4 * a)

theorem parabola_focus :
  ∃ (h k : ℝ), (∀ x y, parabola_equation x y ↔ y - k = -1/8 * (x - h)^2) ∧
  focus h k (p_value (-1/8)) = (0, -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l755_75514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_perimeter_l755_75594

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem triangle_minimum_perimeter (l m n : ℕ) (hl : l > m) (hm : m > n) 
  (h_frac : fractional_part (3^l / 10000 : ℝ) = fractional_part (3^m / 10000 : ℝ) ∧ 
            fractional_part (3^m / 10000 : ℝ) = fractional_part (3^n / 10000 : ℝ)) :
  (∀ l' m' n' : ℕ, l' > m' → m' > n' → 
    fractional_part (3^l' / 10000 : ℝ) = fractional_part (3^m' / 10000 : ℝ) ∧ 
    fractional_part (3^m' / 10000 : ℝ) = fractional_part (3^n' / 10000 : ℝ) →
    l' + m' + n' ≥ 3003) ∧
  (∃ l' m' n' : ℕ, l' > m' ∧ m' > n' ∧
    fractional_part (3^l' / 10000 : ℝ) = fractional_part (3^m' / 10000 : ℝ) ∧
    fractional_part (3^m' / 10000 : ℝ) = fractional_part (3^n' / 10000 : ℝ) ∧
    l' + m' + n' = 3003) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_perimeter_l755_75594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l755_75503

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2)
  (h4 : Real.sin α = 3/5) (h5 : Real.cos (α - β) = 12/13) : 
  Real.sin β = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l755_75503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l755_75577

def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l755_75577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pairs_count_l755_75505

def count_pairs : ℕ := 
  Finset.sum (Finset.range 5) (λ m => 
    Finset.card (Finset.filter (λ n => 0 < n ∧ n < 30 - (m + 1)^2) (Finset.range 30)))

theorem pairs_count : count_pairs = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pairs_count_l755_75505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminium_hydroxide_weight_l755_75518

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The molecular weight of Aluminium hydroxide in g/mol -/
def molecular_weight_AlOH3 : ℝ := atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

/-- Theorem stating that the molecular weight of Aluminium hydroxide is approximately 78.01 g/mol -/
theorem aluminium_hydroxide_weight : 
  |molecular_weight_AlOH3 - 78.01| < 0.01 := by
  -- Unfold the definition of molecular_weight_AlOH3
  unfold molecular_weight_AlOH3
  -- Simplify the expression
  simp [atomic_weight_Al, atomic_weight_O, atomic_weight_H]
  -- The proof itself is omitted for brevity
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminium_hydroxide_weight_l755_75518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cross_section_area_l755_75578

/-- The area of a cross-section of a regular quadrilateral pyramid -/
theorem pyramid_cross_section_area (a : ℝ) (α : ℝ) 
  (h1 : a > 0) 
  (h2 : 0 < α)
  (h3 : α ≤ Real.arctan (Real.sqrt 2 / 2)) :
  let cross_section_area := (a^2 / 8) * Real.cos α * (1 / Real.tan α) * Real.tan (2 * α)
  ∃ (height : ℝ) (lateral_edge : ℝ),
    height > 0 ∧ 
    lateral_edge > 0 ∧
    height * Real.tan α = lateral_edge * Real.cos α ∧
    cross_section_area = 
      (lateral_edge * Real.sin α / 2) * (a * Real.cos α * Real.tan (2 * α) / Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_cross_section_area_l755_75578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_semicircle_area_l755_75598

/-- Predicate for a set representing a semicircle in ℝ² -/
def is_semicircle (s : Set ℝ × Set ℝ) : Prop :=
  sorry

/-- Predicate for a set representing an isosceles right triangle in ℝ² -/
def is_isosceles_right_triangle (t : Set ℝ × Set ℝ × Set ℝ) : Prop :=
  sorry

/-- Predicate indicating that a triangle is inscribed in a semicircle -/
def triangle_inscribed_in_semicircle 
  (t : Set ℝ × Set ℝ × Set ℝ) (s : Set ℝ × Set ℝ) : Prop :=
  sorry

/-- Function returning the leg length of a triangle -/
def triangle_leg_length (t : Set ℝ × Set ℝ × Set ℝ) : ℝ :=
  sorry

/-- Predicate indicating that a triangle's hypotenuse is on the diameter of a semicircle -/
def triangle_hypotenuse_on_diameter 
  (t : Set ℝ × Set ℝ × Set ℝ) (s : Set ℝ × Set ℝ) : Prop :=
  sorry

/-- Function returning the area of a semicircle -/
noncomputable def area (s : Set ℝ × Set ℝ) : ℝ :=
  sorry

/-- The area of a semicircle in which an isosceles right triangle with legs of length 1 
    is inscribed (with its hypotenuse on the diameter) is equal to π/4. -/
theorem isosceles_right_triangle_semicircle_area :
  ∀ (s : Set ℝ × Set ℝ) (t : Set ℝ × Set ℝ × Set ℝ),
    is_semicircle s ∧ 
    is_isosceles_right_triangle t ∧
    triangle_inscribed_in_semicircle t s ∧
    triangle_leg_length t = 1 ∧
    triangle_hypotenuse_on_diameter t s →
    area s = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_semicircle_area_l755_75598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_syrup_used_l755_75587

/-- Represents the sales and syrup usage in a malt shop on a weekend day. -/
structure MaltShopSales where
  shakes : Nat
  cones : Nat
  sundaes : Nat
  shakesSyrup : Float
  conesSyrup : Float
  sundaesSyrup : Float
  complimentaryToppingAmount : Float
  complimentaryToppingPercentage : Float
  discountPercentage : Float
  discountThreshold : Nat

/-- Calculates the total amount of chocolate syrup used given the sales data. -/
def calculateTotalSyrup (sales : MaltShopSales) : Float :=
  let baseSyrup := (sales.shakes.toFloat * sales.shakesSyrup) + 
                   (sales.cones.toFloat * sales.conesSyrup) + 
                   (sales.sundaes.toFloat * sales.sundaesSyrup)
  let complimentaryTopping := 0 -- Simplified as per problem constraints
  let totalItems := sales.shakes + sales.cones + sales.sundaes
  let discount := if totalItems > sales.discountThreshold 
                  then sales.discountPercentage * baseSyrup 
                  else 0
  baseSyrup + complimentaryTopping - discount

/-- Theorem stating that the total syrup used is 86.545 ounces for the given sales data. -/
theorem total_syrup_used (sales : MaltShopSales) 
  (h1 : sales.shakes = 7)
  (h2 : sales.cones = 5)
  (h3 : sales.sundaes = 3)
  (h4 : sales.shakesSyrup = 5.5)
  (h5 : sales.conesSyrup = 8)
  (h6 : sales.sundaesSyrup = 4.2)
  (h7 : sales.complimentaryToppingAmount = 0.3)
  (h8 : sales.complimentaryToppingPercentage = 0.1)
  (h9 : sales.discountPercentage = 0.05)
  (h10 : sales.discountThreshold = 13) :
  calculateTotalSyrup sales = 86.545 := by
  sorry

#eval calculateTotalSyrup {
  shakes := 7,
  cones := 5,
  sundaes := 3,
  shakesSyrup := 5.5,
  conesSyrup := 8,
  sundaesSyrup := 4.2,
  complimentaryToppingAmount := 0.3,
  complimentaryToppingPercentage := 0.1,
  discountPercentage := 0.05,
  discountThreshold := 13
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_syrup_used_l755_75587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_test_probability_l755_75591

/-- The probability of making a single shot -/
def p : ℚ := 2/3

/-- The number of successful shots needed to pass the test -/
def required_successes : ℕ := 3

/-- The maximum number of attempts allowed -/
def max_attempts : ℕ := 5

/-- The probability of passing the basketball shooting test -/
def pass_probability : ℚ := 64/81

/-- Theorem stating that the probability of passing the basketball shooting test is 64/81 -/
theorem basketball_test_probability :
  p^required_successes +
  (Nat.choose (max_attempts - 1) required_successes) * p^required_successes * (1-p)^(max_attempts - required_successes - 1) +
  (Nat.choose max_attempts required_successes) * p^required_successes * (1-p)^(max_attempts - required_successes) =
  pass_probability := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_test_probability_l755_75591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l755_75570

theorem divisibility_in_subset (S : Finset Nat) (h1 : S ⊆ Finset.range 201) (h2 : S.card = 101) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l755_75570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sequence_theorem_l755_75558

/-- The hyperbola x^2 - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The sequence of y-coordinates -/
noncomputable def y_seq : ℕ → ℝ → ℝ
| 0, y₀ => y₀
| n + 1, y₀ => (y_seq n y₀)^2 - 1 / (2 * y_seq n y₀)

/-- The number of starting positions -/
def num_starting_positions : ℕ := 2^2010 - 2

/-- The main theorem -/
theorem hyperbola_sequence_theorem :
  ∀ y₀ : ℝ, y₀ ≠ 0 →
  (∃ k : ℕ, k ∈ Finset.range num_starting_positions ∧
    y₀ = Real.tan (k * Real.pi / (2^2010 - 1))) ↔
  y_seq 2010 y₀ = y₀ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sequence_theorem_l755_75558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_m_less_n_value_l755_75556

/-- Represents the possible labels on the balls -/
inductive Label
  | one
  | two
deriving Repr, DecidableEq

/-- Represents the bag of balls -/
def Bag : Finset Label := sorry

/-- The number of balls labeled 1 -/
def num_ones : ℕ := 4

/-- The number of balls labeled 2 -/
def num_twos : ℕ := 2

/-- The total number of balls -/
def total_balls : ℕ := 6

/-- Axiom: The bag contains the correct number of balls with each label -/
axiom bag_composition : (Bag.filter (· = Label.one)).card = num_ones ∧
                        (Bag.filter (· = Label.two)).card = num_twos ∧
                        Bag.card = total_balls

/-- The probability of drawing any specific ball -/
def prob_draw_ball : ℚ := 1 / total_balls

/-- The sum of two drawn balls -/
def sum_draw : Label → Label → ℕ
  | Label.one, Label.one => 2
  | Label.one, Label.two => 3
  | Label.two, Label.one => 3
  | Label.two, Label.two => 4

/-- The probability distribution of the sum of two drawn balls -/
noncomputable def prob_sum (n : ℕ) : ℚ :=
  match n with
  | 2 => 1 / 15
  | 3 => 4 / 15
  | 4 => 1 / 15
  | _ => 0

/-- The probability that m < n -/
noncomputable def prob_m_less_n : ℚ :=
  (1 / 15) * (14 / 15) + (4 / 15) * (11 / 15)

/-- Main theorem: The probability that m < n is 26/75 -/
theorem prob_m_less_n_value : prob_m_less_n = 26 / 75 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_m_less_n_value_l755_75556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wages_of_men_l755_75588

/-- Represents the daily wage of a worker -/
structure Wage where
  amount : ℝ
  mk_wage : amount > 0

/-- Represents a group of workers -/
structure WorkerGroup where
  count : ℕ
  wage : Wage

/-- Calculates the total wages for a group of workers -/
def totalWages (group : WorkerGroup) : ℝ :=
  group.count * group.wage.amount

/-- The problem setup -/
structure ProblemSetup where
  men : WorkerGroup
  women : WorkerGroup
  boys : WorkerGroup
  men_count : men.count = 12
  boys_count : boys.count = 18
  equal_work_men_women : totalWages men = totalWages women
  equal_work_women_boys : totalWages women = totalWages boys
  total_earnings : totalWages men + totalWages women + totalWages boys = 320

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

notation:50 x " ≈ " y => approx_equal x y 0.01

theorem wages_of_men (setup : ProblemSetup) :
  approx_equal (totalWages setup.men) 106.68 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wages_of_men_l755_75588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l755_75572

/-- Given an ellipse and a chord bisected by a point, prove the equation of the line on which the chord lies. -/
theorem chord_equation (x y : ℝ) (h : (x^2 / 36) + (y^2 / 9) = 1) 
  (bisect_point : ℝ × ℝ) (hbisect : bisect_point = (4, 2)) : 
  ∃ (a b c : ℝ), a*x + b*y + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l755_75572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_neg_cos_l755_75593

noncomputable def f (x : ℝ) : ℝ := Real.sin ((1/5) * x + (13/6) * Real.pi)

noncomputable def g (x : ℝ) : ℝ := f (x - (10/3) * Real.pi)

theorem g_equals_neg_cos (x : ℝ) : g x = -Real.cos ((1/5) * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_neg_cos_l755_75593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_not_simple_l755_75583

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  depth : ℝ

/-- Represents a cube -/
structure Cube where
  side : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.depth

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.side^3

/-- Calculates the ratio of liquid level rise in two cones when a cube is dropped in each -/
noncomputable def liquidLevelRiseRatio (cone1 cone2 : Cone) (cube : Cube) : ℝ := 
  let v1 := coneVolume cone1
  let v2 := coneVolume cone2
  let vc := cubeVolume cube
  let x := (v1 + vc) / v1
  let y := (v2 + vc) / v2
  (cone1.depth * (x^(1/3) - 1)) / (cone2.depth * (y^(1/3) - 1))

theorem liquid_level_rise_ratio_not_simple 
  (cone1 : Cone) 
  (cone2 : Cone) 
  (cube : Cube) 
  (h1 : cone1.radius = 5) 
  (h2 : cone1.depth = 5) 
  (h3 : cone2.radius = 10) 
  (h4 : cone2.depth = 10) 
  (h5 : cube.side = 2) : 
  ∃ (r : ℝ), liquidLevelRiseRatio cone1 cone2 cube = r ∧ 
  ¬(∃ (a b : ℤ), r = a / b) ∧ 
  ¬(∃ (n : ℕ) (x : ℝ), x^n = r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_not_simple_l755_75583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_k_l755_75545

noncomputable section

variable (a : ℝ)
variable (ha : 0 < a ∧ a ≤ 4)

def f (a : ℝ) (x : ℝ) : ℝ := 2*x - 2/x^2 + a/x

theorem characterization_of_k (a : ℝ) (ha : 0 < a ∧ a ≤ 4) (k : ℝ) : 
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → |f a x₁ - f a x₂| > k * |x₁ - x₂|) ↔ 
  k ≤ 2 - a^3 / 108 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_k_l755_75545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_after_changes_l755_75581

theorem original_price_after_changes (p q d : ℝ) (hp : p ≥ 0) (hq : q ≥ 0) (hd : d > 0) :
  let final_price := λ x : ℝ ↦ x * (1 + p / 100) * (1 - q / 100)
  ∃ x : ℝ, x > 0 ∧ final_price x = d ∧ x = 100 * d / (100 + p - q - p * q / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_after_changes_l755_75581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l755_75560

noncomputable def sequence_term (n : ℕ) : ℝ := 3 + 2 * (n - 1 : ℝ)

noncomputable def sum_of_terms (n : ℕ) : ℝ := n^2 + 2*n

noncomputable def expression (n : ℕ) : ℝ := (sum_of_terms n + 33) / n

theorem min_value_of_expression :
  ∀ n : ℕ, n > 0 → expression n ≥ 13.5 ∧ ∃ m : ℕ, m > 0 ∧ expression m = 13.5 :=
by
  sorry

#check min_value_of_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l755_75560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_properties_l755_75511

/-- Represents a 192-digit number as a list of digits -/
def Digits := List Nat

/-- Represents a 92-digit number as a list of digits -/
def Digits92 := List Nat

/-- Check if a number can be obtained from another by deleting digits -/
def can_obtain_by_deleting (original : Digits) (result : Digits92) : Prop := sorry

/-- The original 192-digit number N -/
def N : Digits := sorry

/-- The largest 92-digit number M obtained from N by deleting 100 digits -/
def M : Digits92 := sorry

/-- The smallest valid 92-digit number m obtained from N by deleting 100 digits -/
def m : Digits92 := sorry

/-- Convert a list of digits to a natural number -/
def to_nat (digits : List Nat) : Nat := sorry

/-- Convert a natural number to a list of digits -/
def to_digits (n : Nat) : List Nat := sorry

theorem number_properties :
  let d := to_nat M - to_nat m
  ∃ k : ℕ, d = 24 * k ∧
  can_obtain_by_deleting N (to_digits (to_nat M - d / 6)) ∧
  can_obtain_by_deleting N (to_digits (to_nat m + d / 8)) ∧
  can_obtain_by_deleting N (to_digits (to_nat M - d / 8)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_properties_l755_75511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_101_l755_75597

theorem binomial_sum_101 : 
  (Finset.sum (Finset.range 51) (fun k => (-1 : ℤ)^k * (Nat.choose 101 (2*k) : ℤ))) = -(2^50 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_101_l755_75597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l755_75528

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (m, 1)
def vector_b : ℝ × ℝ := (2, -6)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem vector_difference_magnitude (m : ℝ) :
  perpendicular (vector_a m) vector_b →
  magnitude (vector_a m - vector_b) = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l755_75528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_percentage_l755_75516

/-- The exchange rate from yen to dollars -/
noncomputable def yen_to_dollar : ℝ := 0.0075

/-- Diana's money in dollars -/
noncomputable def diana_money : ℝ := 500

/-- Etienne's money in yen -/
noncomputable def etienne_money_yen : ℝ := 5000

/-- Etienne's money converted to dollars -/
noncomputable def etienne_money_dollar : ℝ := etienne_money_yen * yen_to_dollar

/-- The percentage difference between two values -/
noncomputable def percentage_difference (old_value new_value : ℝ) : ℝ :=
  ((old_value - new_value) / old_value) * 100

theorem money_difference_percentage :
  percentage_difference diana_money etienne_money_dollar = 92.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_money_difference_percentage_l755_75516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_theorem_l755_75539

/-- Calculates the gross profit percent when an item is sold with a discount -/
noncomputable def gross_profit_percent_with_discount (cost : ℝ) (discount_rate : ℝ) (profit_rate_without_discount : ℝ) : ℝ :=
  let selling_price_without_discount := cost * (1 + profit_rate_without_discount)
  let selling_price_with_discount := selling_price_without_discount * (1 - discount_rate)
  let profit_with_discount := selling_price_with_discount - cost
  (profit_with_discount / cost) * 100

/-- 
Theorem: If an item is sold at a 10% discount and the gross profit without discount 
would have been 30% of the cost, then the gross profit percent with the discount is 17%.
-/
theorem discount_profit_theorem (cost : ℝ) (cost_positive : cost > 0) :
  gross_profit_percent_with_discount cost 0.1 0.3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_theorem_l755_75539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_equal_digits_in_prime_power_l755_75582

theorem consecutive_equal_digits_in_prime_power (p : ℕ) (n : ℕ) 
  (hp : Nat.Prime p) (hn : n ≥ 2) :
  ∃ k : ℕ, k > 0 ∧ ∃ m : ℕ, m > 0 ∧ 
    ∃ r : ℕ, r < 10^m ∧ p^k = (10^n - 1) / 9 * 10^m + r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_equal_digits_in_prime_power_l755_75582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l755_75515

theorem triangle_properties (a b c A B C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0)
  (h_angles : A + B + C = Real.pi)
  (h_sides : a = 2 * Real.sin A ∧ b = 2 * Real.sin B ∧ c = 2 * Real.sin C)
  (h_equation : (Real.sin B * Real.sin C) / (3 * Real.sin A) = (Real.cos A) / a + (Real.cos C) / c)
  (h_area : (a^2 + b^2 - c^2) / 4 = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  (∃ R : ℝ, R = 2 ∧ R * Real.sin A = a / 2 ∧ R * Real.sin B = b / 2 ∧ R * Real.sin C = c / 2) ∧
  (1/2 ≤ c / (a + b) ∧ c / (a + b) < 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l755_75515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_solution_l755_75543

-- Define the polynomial type
def MyPolynomial (α : Type*) := α → α

-- Define the identity that P(x) must satisfy
def SatisfiesIdentity (P : MyPolynomial ℝ) : Prop :=
  ∀ x : ℝ, (x - 1) * P (x + 1) = (x + 2) * P x

-- State the theorem
theorem polynomial_identity_solution :
  ∀ P : MyPolynomial ℝ, SatisfiesIdentity P →
  ∃ a : ℝ, a ≠ 0 ∧ (∀ x : ℝ, P x = a * (x^3 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_solution_l755_75543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_perpendicular_m_l755_75575

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := 
  if l.b ≠ 0 then -l.a / l.b else 0  -- Handle vertical lines

/-- Perpendicularity of two lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The given lines -/
def l : Line := ⟨1, 1, -1⟩
def m : Line := ⟨1, -1, 1⟩

/-- Theorem: l is perpendicular to m -/
theorem l_perpendicular_m : perpendicular l m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_perpendicular_m_l755_75575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_value_l755_75526

theorem cos_two_theta_value (θ : Real) (h : Real.sin (π - θ) = 1/3) : Real.cos (2*θ) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_value_l755_75526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_green_apples_l755_75522

def total_apples : ℕ := 9
def green_apples : ℕ := 4

theorem probability_two_green_apples :
  (Nat.choose green_apples 2 : ℚ) / (Nat.choose total_apples 2) = 1/6 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_green_apples_l755_75522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l755_75547

def word : String := "balloon"
def total_letters : Nat := 7
def repeated_letter : Char := 'l'
def repeated_count : Nat := 2

theorem balloon_arrangements :
  (Nat.factorial total_letters) / (Nat.factorial repeated_count) = 2520 := by
  -- Proof steps would go here
  sorry

#eval (Nat.factorial total_letters) / (Nat.factorial repeated_count)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_l755_75547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_product_l755_75521

theorem quadratic_roots_product (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ - 3 = 0 → x₂^2 - x₂ - 3 = 0 → 
  (x₁^5 - 20) * (3 * x₂^4 - 2 * x₂ - 35) = -1063 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_product_l755_75521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_exists_l755_75551

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a line segment on the grid -/
structure GridSegment where
  start : GridPoint
  finish : GridPoint

/-- Represents a division of the square -/
structure SquareDivision where
  segments : List GridSegment

def square_side_length : ℕ := 5

/-- Calculates the area of a part in the division -/
noncomputable def area_of_part (division : SquareDivision) : ℚ :=
  sorry

/-- Calculates the total length of segments in the division -/
def total_segment_length (division : SquareDivision) : ℕ :=
  sorry

/-- Checks if all parts have equal area -/
def has_equal_parts (division : SquareDivision) : Prop :=
  ∀ part₁ part₂, area_of_part part₁ = area_of_part part₂

/-- Checks if the division uses only grid lines -/
def uses_only_grid_lines (division : SquareDivision) : Prop :=
  ∀ segment ∈ division.segments, 
    (segment.start.x = segment.finish.x ∨ segment.start.y = segment.finish.y)

theorem square_division_exists : 
  ∃ (division : SquareDivision),
    division.segments.length = 5 ∧
    has_equal_parts division ∧
    uses_only_grid_lines division ∧
    total_segment_length division = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_exists_l755_75551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon1_best_at_209_95_l755_75568

noncomputable def coupon1_discount (price : ℝ) : ℝ := 0.12 * price

noncomputable def coupon2_discount (price : ℝ) : ℝ := 
  if price ≥ 120 then 25 else 0

noncomputable def coupon3_discount (price : ℝ) : ℝ := 
  if price > 150 then 0.20 * (price - 150) else 0

theorem coupon1_best_at_209_95 : 
  let price : ℝ := 209.95
  coupon1_discount price > coupon2_discount price ∧ 
  coupon1_discount price > coupon3_discount price := by
  -- Unfold definitions
  unfold coupon1_discount coupon2_discount coupon3_discount
  -- Split into two goals
  constructor
  -- Prove coupon1 > coupon2
  · norm_num
  -- Prove coupon1 > coupon3
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon1_best_at_209_95_l755_75568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l755_75533

/-- The coefficient of x^2 in the expansion of (x + 1/x)^8 is 56 -/
theorem coefficient_x_squared_in_expansion : ∀ x : ℝ, x ≠ 0 →
  (∃ c : ℝ, c = 56 ∧ 
    ∃ f : ℝ → ℝ, (fun y => (y + 1/y)^8) = (fun y => c * y^2 + f y) ∧ 
    (∀ y : ℝ, y ≠ 0 → Filter.Tendsto (fun y => f y / y^2) (Filter.atTop) (nhds 0))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_expansion_l755_75533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juice_P_in_A_l755_75536

/-- Represents the amount of juice P in smoothie A -/
noncomputable def P_A : ℝ := sorry

/-- Represents the amount of juice V in smoothie A -/
noncomputable def V_A : ℝ := sorry

/-- Represents the amount of juice P in smoothie Y -/
noncomputable def P_Y : ℝ := sorry

/-- Represents the amount of juice V in smoothie Y -/
noncomputable def V_Y : ℝ := sorry

/-- The total amount of juice P is 24 oz -/
axiom total_P : P_A + P_Y = 24

/-- The total amount of juice V is 25 oz -/
axiom total_V : V_A + V_Y = 25

/-- The ratio of P to V in smoothie A is 4:1 -/
axiom ratio_A : P_A / V_A = 4 / 1

/-- The ratio of P to V in smoothie Y is 1:5 -/
axiom ratio_Y : P_Y / V_Y = 1 / 5

/-- Theorem: The amount of juice P in smoothie A is 20 oz -/
theorem juice_P_in_A : P_A = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juice_P_in_A_l755_75536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_1_inequality_2_l755_75567

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom g_odd : ∀ x : ℝ, g (-x) = -g x
axiom f_increasing : ∀ x y : ℝ, 0 ≤ x → x < y → f x < f y
axiom g_increasing : ∀ x y : ℝ, x < y → g x < g y

-- State the theorems to be proved
theorem inequality_1 : g (Real.sin (π/5)) < g ((2:ℝ)^(1/10)) := by sorry

theorem inequality_2 : g (f 1) < g (f 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_1_inequality_2_l755_75567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_prime_power_solutions_l755_75553

def f (x : ℤ) : ℤ := 2 * x^2 + x - 6

def is_prime_power (n : ℤ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = Int.ofNat (p^k)

theorem quadratic_prime_power_solutions :
  ∀ x : ℤ, is_prime_power (f x) ↔ x = -3 ∨ x = 2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_prime_power_solutions_l755_75553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_age_l755_75546

def Sibling := Nat

def is_multiple_of_4 (n : Nat) : Prop := ∃ k, n = 4 * k

theorem sam_age (ages : Finset Nat) (sam : Nat) :
  ages.card = 5 ∧
  ages = {4, 6, 8, 10, 12} ∧
  (∃ (a b : Nat), a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a + b = 18) ∧
  (∃ (c d : Nat), c ∈ ages ∧ d ∈ ages ∧ c ≠ d ∧ is_multiple_of_4 c ∧ is_multiple_of_4 d) ∧
  (sam ∈ ages ∧ 6 ∈ ages ∧ sam ≠ 6) →
  sam = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_age_l755_75546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l755_75596

theorem negation_of_proposition :
  (¬∀ x : ℝ, x > 0 → x - Real.log x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - Real.log x₀ ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l755_75596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l755_75542

noncomputable def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let D : ℝ × ℝ := (0, b)
  let F1 : ℝ × ℝ := (-c, 0)
  let F2 : ℝ × ℝ := (c, 0)
  let A : ℝ × ℝ := (-a, 0)
  3 * (F1 - D) = (A - D) + 2 * (F2 - D) →
  Eccentricity a b = 1/5 := by
  sorry

#check ellipse_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l755_75542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l755_75523

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 2 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(√2 / 2)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l755_75523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l755_75510

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the given parallel lines is 4√13/13 -/
theorem distance_between_given_lines :
  let line1 := λ (x y : ℝ) => 2*x + 3*y - 3 = 0
  let line2 := λ (x y : ℝ) => 4*x + 6*y + 2 = 0
  ∃ (m : ℝ), (∀ x y, line2 x y ↔ 4*x + m*y + 2 = 0) ∧
  distance_between_parallel_lines 4 6 (-6) 2 = 4 * Real.sqrt 13 / 13 :=
by
  sorry

#check distance_between_given_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l755_75510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_2_l755_75554

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - x - 2
  else x / (x + 4) + Real.log (abs x) / Real.log 4

theorem f_composition_at_2 : f (f 2) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_2_l755_75554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_one_l755_75552

/-- The largest prime with 2023 digits -/
noncomputable def p : ℕ := sorry

/-- p is prime -/
axiom p_prime : Nat.Prime p

/-- p has 2023 digits -/
axiom p_digits : Nat.log 10 p + 1 = 2023

/-- p is the largest prime with 2023 digits -/
axiom p_largest : ∀ q : ℕ, Nat.Prime q → Nat.log 10 q + 1 = 2023 → q ≤ p

/-- The smallest positive integer k such that p^2 - k is divisible by 24 -/
noncomputable def k : ℕ := sorry

/-- k is positive -/
axiom k_pos : k > 0

/-- p^2 - k is divisible by 24 -/
axiom div_24 : 24 ∣ (p^2 - k)

/-- k is the smallest positive integer satisfying the condition -/
axiom k_smallest : ∀ m : ℕ, m > 0 → 24 ∣ (p^2 - m) → k ≤ m

theorem smallest_k_is_one : k = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_one_l755_75552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_less_than_50_degrees_l755_75589

theorem angle_less_than_50_degrees :
  ∃ x : ℝ, Real.sin x = Real.sqrt 3 / 2 - 1 / 10 ∧ x < 50 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_less_than_50_degrees_l755_75589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l755_75530

noncomputable def f (x : ℝ) := x^2 - Real.log (2*x)

theorem f_monotone_decreasing :
  StrictMonoOn f (Set.Ioc (0 : ℝ) (Real.sqrt 2 / 2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l755_75530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l755_75540

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * log x

-- State the theorem
theorem max_m_value (m : ℝ) :
  (∀ x : ℝ, x > 0 → f x ≥ (-x^2 + m*x - 3) / 2) →
  m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l755_75540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_edges_non_isosceles_l755_75580

/-- A tetrahedron is a polyhedron with 4 faces and 6 edges. -/
structure Tetrahedron where
  edges : Fin 6 → ℝ
  faces : Fin 4 → Fin 3 → Fin 6

/-- A face is isosceles if at least two of its edges have the same length. -/
def is_isosceles (t : Tetrahedron) (face : Fin 4) : Prop :=
  ∃ (i j : Fin 3), i ≠ j ∧ t.edges (t.faces face i) = t.edges (t.faces face j)

/-- The number of distinct edge lengths in a tetrahedron. -/
noncomputable def distinct_edge_count (t : Tetrahedron) : ℕ :=
  Finset.card (Finset.image t.edges Finset.univ)

/-- Main theorem: In a tetrahedron where no face is isosceles, 
    the minimum number of edges with different lengths is 3. -/
theorem min_distinct_edges_non_isosceles (t : Tetrahedron) 
  (h : ∀ face : Fin 4, ¬ is_isosceles t face) : 
  distinct_edge_count t ≥ 3 ∧ 
  ∃ t', (∀ face : Fin 4, ¬ is_isosceles t' face) ∧ distinct_edge_count t' = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distinct_edges_non_isosceles_l755_75580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l755_75509

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 + Real.sin x

-- Define the interval
def interval : Set ℝ := {x | 2 * Real.pi / 5 ≤ x ∧ x ≤ 3 * Real.pi / 4}

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = (1 + Real.sqrt 2) / 2 ∧
  ∀ (y : ℝ), y ∈ interval → f y ≤ (1 + Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l755_75509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_irrational_l755_75544

theorem cos_theta_irrational (p q : ℕ) (θ : ℝ) : 
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    Real.cos (p * θ) = (b^2 + c^2 - a^2) / (2 * b * c) ∧
    Real.cos (q * θ) = (a^2 + c^2 - b^2) / (2 * a * c)) →
  Nat.Coprime p q →
  Irrational (Real.cos θ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_irrational_l755_75544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_problem_l755_75519

theorem digit_sum_problem (a b c : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧  -- a, b, c are digits
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧     -- a, b, c are distinct
  (10*a + b) + (10*a + c) + (10*b + c) +
  (10*b + a) + (10*c + a) + (10*c + b) = 528 -- sum condition
  →
  ({a, b, c} : Finset ℕ) = {7, 8, 9} :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_problem_l755_75519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_property_l755_75550

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A₁ : ℝ × ℝ
  B₁ : ℝ × ℝ
  C₁ : ℝ × ℝ
  h_midpoint_A₁ : A₁ = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  h_midpoint_B₁ : B₁ = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  h_midpoint_C₁ : C₁ = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem median_property (t : Triangle) (h : distance t.A t.A₁ = 3/2 * distance t.B t.C) :
  (distance t.A t.A₁)^2 = (distance t.B t.B₁)^2 + (distance t.C t.C₁)^2 := by
  sorry

#check median_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_property_l755_75550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l755_75584

/-- Represents a hyperbola with equation x²/a² - y²/3 = 1 -/
structure Hyperbola where
  a : ℝ
  h_a_pos : a > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (h.a^2 + 3) / h.a

/-- An asymptote equation of a hyperbola -/
def asymptote_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => Real.sqrt 3 * x - y = 0

/-- Theorem stating that there exists an asymptote for the hyperbola with eccentricity 2 -/
theorem hyperbola_asymptote (h : Hyperbola) (h_ecc : eccentricity h = 2) :
  ∃ x y, asymptote_equation h x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l755_75584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_leaked_water_is_600_l755_75559

/-- The rate at which the largest hole leaks water, in ounces per minute -/
noncomputable def largest_hole_rate : ℝ := 3

/-- The rate at which the medium-sized hole leaks water, in ounces per minute -/
noncomputable def medium_hole_rate : ℝ := largest_hole_rate / 2

/-- The rate at which the smallest hole leaks water, in ounces per minute -/
noncomputable def smallest_hole_rate : ℝ := medium_hole_rate / 3

/-- The duration of the rain, in minutes -/
noncomputable def rain_duration : ℝ := 2 * 60

/-- The total amount of water leaked from all three holes during the rain -/
noncomputable def total_leaked_water : ℝ := (largest_hole_rate + medium_hole_rate + smallest_hole_rate) * rain_duration

theorem total_leaked_water_is_600 : total_leaked_water = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_leaked_water_is_600_l755_75559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l755_75576

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

-- Define the perimeter of a triangle
noncomputable def TrianglePerimeter (p q r : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) +
  Real.sqrt ((q.1 - r.1)^2 + (q.2 - r.2)^2) +
  Real.sqrt ((r.1 - p.1)^2 + (r.2 - p.2)^2)

-- Define the length of a line segment
noncomputable def SegmentLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Main theorem
theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (0, 2) ∈ Ellipse a b)
  (Q : ℝ × ℝ) (h4 : Q ∈ Ellipse a b)
  (h5 : Q ≠ (a, 0) ∧ Q ≠ (-a, 0))
  (h6 : TrianglePerimeter Q (-c, 0) (c, 0) = 4 + 4 * Real.sqrt 2)
  (k1 k2 : ℝ) (A B C D : ℝ × ℝ)
  (h7 : A ∈ Ellipse a b ∧ B ∈ Ellipse a b)
  (h8 : C ∈ Ellipse a b ∧ D ∈ Ellipse a b)
  (h9 : A.2 = k1 * (A.1 + c) ∧ B.2 = k1 * (B.1 + c))
  (h10 : C.2 = k2 * (C.1 - c) ∧ D.2 = k2 * (D.1 - c))
  (h11 : SegmentLength A B + SegmentLength C D = 6 * Real.sqrt 2)
  (c : ℝ) (h12 : a^2 = b^2 + c^2) :
  (a = 2 * Real.sqrt 2 ∧ b = 2) ∧ (k1 * k2 = 1/2 ∨ k1 * k2 = -1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l755_75576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_decreasing_interval_l755_75504

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (a-2)x^2 + (a-1)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^2 + (a - 1) * x + 3

/-- The decreasing interval of a function -/
def DecreasingInterval (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x > f y

theorem even_function_decreasing_interval (a : ℝ) :
  IsEven (f a) → DecreasingInterval (f a) (Set.Ici 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_decreasing_interval_l755_75504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_plus_2y_equals_one_l755_75531

theorem cos_x_plus_2y_equals_one (x y a : ℝ) 
  (hx : x ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (hy : y ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_plus_2y_equals_one_l755_75531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_min_value_sum_min_value_sum_exact_l755_75500

-- Define the function f
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

-- Part I: Prove the solution set of f(x + 3/2) ≥ 0
theorem solution_set_f (x : ℝ) : 
  f (x + 3/2) ≥ 0 ↔ x ∈ Set.Icc (-2) 2 := by sorry

-- Part II: Prove the minimum value of 3p + 2q + r
theorem min_value_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 := by sorry

-- Prove that 9/4 is indeed the minimum value
theorem min_value_sum_exact (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  ∃ (p' q' r' : ℝ), p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ 
    1/(3*p') + 1/(2*q') + 1/r' = 4 ∧ 
    3*p' + 2*q' + r' = 9/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_min_value_sum_min_value_sum_exact_l755_75500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jebb_take_home_pay_l755_75590

noncomputable def gross_salary : ℚ := 6500

noncomputable def federal_tax (income : ℚ) : ℚ :=
  (10 / 100) * min income 2000 +
  (15 / 100) * max (min income 4000 - 2000) 0 +
  (25 / 100) * max (income - 4000) 0

def health_insurance : ℚ := 300

noncomputable def retirement_contribution (salary : ℚ) : ℚ := (7 / 100) * salary

noncomputable def total_deductions (salary : ℚ) : ℚ :=
  federal_tax salary + health_insurance + retirement_contribution salary

noncomputable def take_home_pay (salary : ℚ) : ℚ :=
  salary - total_deductions salary

theorem jebb_take_home_pay :
  take_home_pay gross_salary = 4620 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jebb_take_home_pay_l755_75590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_increase_values_l755_75507

/-- The surface area of a cylinder with radius R and height H -/
noncomputable def cylinderSurfaceArea (R H : ℝ) : ℝ := 2 * Real.pi * R^2 + 2 * Real.pi * R * H

/-- The condition for equal surface area increase -/
def equalSurfaceAreaIncrease (x : ℝ) : Prop :=
  cylinderSurfaceArea (10 + x) (5 + x) - cylinderSurfaceArea 10 5 =
  cylinderSurfaceArea 10 (5 + x) - cylinderSurfaceArea 10 5

/-- Theorem: The values of x that satisfy the equal surface area increase condition -/
theorem cylinder_surface_area_increase_values :
  ∃ (x₁ x₂ : ℝ), x₁ = -10 + 5 * Real.sqrt 6 ∧ 
                 x₂ = -10 - 5 * Real.sqrt 6 ∧
                 equalSurfaceAreaIncrease x₁ ∧
                 equalSurfaceAreaIncrease x₂ ∧
                 ∀ (x : ℝ), equalSurfaceAreaIncrease x → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_increase_values_l755_75507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_permission_slips_l755_75524

theorem scout_permission_slips (total_scouts : ℝ) (h1 : total_scouts > 0) : 
  (0.50 * (0.45 * total_scouts) + 0.6818 * (total_scouts - 0.45 * total_scouts)) / total_scouts = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_permission_slips_l755_75524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_even_solution_l755_75520

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem sin_shift_even_solution :
  ∃ m : ℝ, m = Real.pi / 12 ∧ is_even (fun x ↦ f (x + m)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_even_solution_l755_75520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_sum_l755_75549

theorem nth_term_sum (n : ℕ) : n^2 + n^3 + (-2 : ℤ) * (n^2 : ℤ) = n^3 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_term_sum_l755_75549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_positive_integers_l755_75538

/-- Definition of set A -/
def A : Set ℕ := {n | ∃ k : ℕ, n = k * Nat.factorial k + k}

/-- Definition of set B as complement of A in ℕ+ -/
def B : Set ℕ := {n | n > 0 ∧ n ∉ A}

/-- Theorem stating the partition properties -/
theorem partition_positive_integers :
  (∀ n : ℕ, n > 0 → (n ∈ A ∨ n ∈ B)) ∧
  (A ∩ B = ∅) ∧
  (∀ a b c : ℕ, a ∈ A → b ∈ A → c ∈ A → b - a = c - b → (a = b ∨ b = c)) ∧
  (∀ a d : ℕ, d > 0 → (∃ N : ℕ, a + N * d ∉ B)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_positive_integers_l755_75538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l755_75529

theorem complex_equation_solution (x y : ℝ) :
  (x / (1 - Complex.I)) + (y / (1 - 2 * Complex.I)) = 5 / (1 - 3 * Complex.I) →
  x = -1 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l755_75529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceva_menelaus_theorem_l755_75599

-- Define the basic geometric objects
variable (A B C A₁ B₁ C₁ A₂ B₂ C₂ : ℝ × ℝ)

-- Define the condition that three points are collinear
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

-- Define the condition that three lines intersect at one point
def intersectAtOnePoint (L₁ L₂ L₃ : Set (ℝ × ℝ)) : Prop := sorry

-- Define the condition that three lines are parallel or intersect at one point
def parallelOrIntersect (L₁ L₂ L₃ : Set (ℝ × ℝ)) : Prop := sorry

-- Define lines as sets of points
def line (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem ceva_menelaus_theorem 
  (h₁ : collinear A B C)
  (h₂ : collinear A₁ B₁ C₁)
  (h₃ : intersectAtOnePoint (line A A₁) (line B B₁) (line C C₁))
  (h₄ : intersectAtOnePoint (line A₁ A₂) (line B₁ B₂) (line C₁ C₂)) :
  parallelOrIntersect (line A A₂) (line B B₂) (line C C₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceva_menelaus_theorem_l755_75599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l755_75508

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  BD : Real

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  t.b * Real.sin t.A + Real.sqrt 3 * t.a * Real.cos t.B = 0 ∧
  t.a = 2 ∧
  t.BD = Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : TriangleProperties t) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1 / 2 : Real) * t.a * t.c * Real.sin t.B = Real.sqrt 3 * 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l755_75508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integral_roots_l755_75565

-- Define the equations
def equation1 (x : ℝ) : Prop := 5 * x^2 + 3 = 40
def equation2 (x : ℝ) : Prop := (3*x - 2)^3 = (x - 2)^3 - 27
def equation3 (x : ℝ) : Prop := (x^2 - 4) = (3*x - 4)

-- Define what it means for a real number to be an integer
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = ↑n

-- Theorem statement
theorem no_integral_roots :
  ¬∃ x : ℝ, (isInteger x ∧ (equation1 x ∨ equation2 x ∨ equation3 x)) := by
  sorry

#check no_integral_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integral_roots_l755_75565
