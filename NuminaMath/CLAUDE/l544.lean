import Mathlib

namespace rogers_retirement_experience_l544_54445

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.peter = 12 ∧
  e.tom = 2 * e.robert ∧
  e.robert = e.peter - 4 ∧
  e.robert = e.mike + 2

/-- The theorem to be proved -/
theorem rogers_retirement_experience (e : Experience) :
  satisfies_conditions e → e.roger + 8 = 50 :=
by
  sorry

end rogers_retirement_experience_l544_54445


namespace interest_difference_l544_54442

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proves that the difference between the principal and the simple interest is 1260 -/
theorem interest_difference :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℝ := 4
  principal - simple_interest principal rate time = 1260 := by
  sorry

end interest_difference_l544_54442


namespace sequence_ratio_l544_54420

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₁ = 1 + d ∧ a₂ = 1 + 2*d ∧ 3 = 1 + 3*d

-- Define the geometric sequence
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ = 1 * r ∧ b₂ = 1 * r^2 ∧ b₃ = 1 * r^3 ∧ 4 = 1 * r^4

theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h1 : arithmetic_sequence a₁ a₂) 
  (h2 : geometric_sequence b₁ b₂ b₃) : 
  (a₁ + a₂) / b₂ = 2 := by
  sorry

end sequence_ratio_l544_54420


namespace final_state_only_beads_l544_54488

/-- Represents the types of items in the exchange system -/
inductive Item
  | Gold
  | Pearl
  | Bead

/-- Represents the state of items in the exchange system -/
structure ItemState :=
  (gold : ℕ)
  (pearl : ℕ)
  (bead : ℕ)

/-- Represents an exchange rule -/
structure ExchangeRule :=
  (input1 : Item)
  (input2 : Item)
  (output : Item)

/-- Applies an exchange rule to the current state -/
def applyExchange (state : ItemState) (rule : ExchangeRule) : ItemState :=
  sorry

/-- Checks if an exchange rule can be applied to the current state -/
def canApplyExchange (state : ItemState) (rule : ExchangeRule) : Prop :=
  sorry

/-- Represents the exchange system with initial state and rules -/
structure ExchangeSystem :=
  (initialState : ItemState)
  (rules : List ExchangeRule)

/-- Defines the final state after all possible exchanges -/
def finalState (system : ExchangeSystem) : ItemState :=
  sorry

/-- Theorem: The final state after all exchanges will only have beads -/
theorem final_state_only_beads (system : ExchangeSystem) :
  system.initialState = ItemState.mk 24 26 25 →
  system.rules = [
    ExchangeRule.mk Item.Gold Item.Pearl Item.Bead,
    ExchangeRule.mk Item.Gold Item.Bead Item.Pearl,
    ExchangeRule.mk Item.Pearl Item.Bead Item.Gold
  ] →
  ∃ n : ℕ, finalState system = ItemState.mk 0 0 n :=
sorry

end final_state_only_beads_l544_54488


namespace sqrt_meaningful_range_l544_54408

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 6) ↔ x ≥ 6 := by sorry

end sqrt_meaningful_range_l544_54408


namespace binomial_expansion_example_l544_54438

theorem binomial_expansion_example : 
  8^4 + 4*(8^3)*2 + 6*(8^2)*(2^2) + 4*8*(2^3) + 2^4 = 10000 := by
sorry

end binomial_expansion_example_l544_54438


namespace fraction_sum_zero_l544_54432

theorem fraction_sum_zero (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end fraction_sum_zero_l544_54432


namespace brand_x_pen_price_l544_54412

/-- The price of a brand X pen satisfies the given conditions -/
theorem brand_x_pen_price :
  ∀ (total_pens brand_x_pens : ℕ) (brand_y_price total_cost brand_x_price : ℚ),
    total_pens = 12 →
    brand_x_pens = 8 →
    brand_y_price = 14/5 →
    total_cost = 40 →
    brand_x_price * brand_x_pens + brand_y_price * (total_pens - brand_x_pens) = total_cost →
    brand_x_price = 18/5 := by
  sorry

end brand_x_pen_price_l544_54412


namespace hyperbola_triangle_area_l544_54480

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point P on the hyperbola
def P : ℝ × ℝ := sorry

-- Define the distances |PF₁| and |PF₂|
def PF₁ : ℝ := sorry
def PF₂ : ℝ := sorry

-- Theorem statement
theorem hyperbola_triangle_area :
  hyperbola P.1 P.2 →
  PF₁ / PF₂ = 3 / 4 →
  (1/2) * ‖F₁ - F₂‖ * ‖P - (F₁ + F₂)/2‖ = 8 * Real.sqrt 5 := by
  sorry

end hyperbola_triangle_area_l544_54480


namespace smaller_number_problem_l544_54484

theorem smaller_number_problem (x y : ℕ) : 
  x * y = 40 → x + y = 14 → min x y = 4 := by
  sorry

end smaller_number_problem_l544_54484


namespace ratio_sum_to_y_l544_54444

theorem ratio_sum_to_y (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 3 / 4) :
  (x + y) / y = 13 / 4 := by
  sorry

end ratio_sum_to_y_l544_54444


namespace estimate_value_l544_54456

theorem estimate_value : 
  3 < (Real.sqrt 3 + 3 * Real.sqrt 2) * Real.sqrt (1/3) ∧ 
  (Real.sqrt 3 + 3 * Real.sqrt 2) * Real.sqrt (1/3) < 4 := by
  sorry

end estimate_value_l544_54456


namespace fixed_point_coordinates_l544_54410

theorem fixed_point_coordinates : ∃! (A : ℝ × ℝ), ∀ (k : ℝ),
  (3 + k) * A.1 + (1 - 2*k) * A.2 + 1 + 5*k = 0 ∧ A = (-1, 2) := by
  sorry

end fixed_point_coordinates_l544_54410


namespace fred_earnings_l544_54417

/-- The amount of money earned given an hourly rate and number of hours worked -/
def moneyEarned (hourlyRate : ℝ) (hoursWorked : ℝ) : ℝ :=
  hourlyRate * hoursWorked

/-- Proof that working 8 hours at $12.5 per hour results in $100 earned -/
theorem fred_earnings : moneyEarned 12.5 8 = 100 := by
  sorry

end fred_earnings_l544_54417


namespace bowtie_equation_l544_54457

-- Define the operation ⊛
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + 1 + Real.sqrt (b + 1 + Real.sqrt (b + 1 + Real.sqrt (b + 1)))))

-- State the theorem
theorem bowtie_equation (h : ℝ) :
  bowtie 5 h = 8 → h = 9 - Real.sqrt 10 := by sorry

end bowtie_equation_l544_54457


namespace esperanza_salary_l544_54419

def gross_salary (rent food mortgage savings taxes : ℚ) : ℚ :=
  rent + food + mortgage + savings + taxes

theorem esperanza_salary :
  let rent : ℚ := 600
  let food : ℚ := (3/5) * rent
  let mortgage : ℚ := 3 * food
  let savings : ℚ := 2000
  let taxes : ℚ := (2/5) * savings
  gross_salary rent food mortgage savings taxes = 4840 := by
  sorry

end esperanza_salary_l544_54419


namespace power_steering_count_l544_54499

theorem power_steering_count (total : ℕ) (power_windows : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 65)
  (h2 : power_windows = 25)
  (h3 : both = 17)
  (h4 : neither = 12) :
  total - neither - (power_windows - both) = 45 :=
by sorry

end power_steering_count_l544_54499


namespace empty_solution_set_iff_a_in_range_l544_54451

theorem empty_solution_set_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ (a ≤ 2 ∨ a ≥ 4) := by
  sorry

end empty_solution_set_iff_a_in_range_l544_54451


namespace derivative_at_one_l544_54450

/-- Given f(x) = 2x³ + x² - 5, prove that f'(1) = 8 -/
theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^3 + x^2 - 5) : 
  deriv f 1 = 8 := by
  sorry

end derivative_at_one_l544_54450


namespace solution_set_a_eq_1_range_of_a_l544_54489

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x - 1| + |a * x - 3 * a|

-- Part 1: Solution set when a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≥ 9/2 ∨ x ≤ -1/2} :=
sorry

-- Part 2: Range of a when solution set is ℝ
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f a x ≥ 5) → a ≥ 2 :=
sorry

end solution_set_a_eq_1_range_of_a_l544_54489


namespace rainfall_water_level_rise_l544_54470

/-- Given 15 liters of rainfall per square meter, the rise in water level in a pool is 1.5 cm. -/
theorem rainfall_water_level_rise :
  let rainfall_per_sqm : ℝ := 15  -- liters per square meter
  let liters_to_cubic_cm : ℝ := 1000  -- 1 liter = 1000 cm³
  let sqm_to_sqcm : ℝ := 10000  -- 1 m² = 10000 cm²
  rainfall_per_sqm * liters_to_cubic_cm / sqm_to_sqcm = 1.5  -- cm
  := by sorry

end rainfall_water_level_rise_l544_54470


namespace trigonometric_identities_l544_54435

theorem trigonometric_identities (α : Real) 
  (h : 3 * Real.sin α - 2 * Real.cos α = 0) : 
  (((Real.cos α - Real.sin α) / (Real.cos α + Real.sin α)) + 
   ((Real.cos α + Real.sin α) / (Real.cos α - Real.sin α)) = 6) ∧ 
  ((Real.sin α)^2 - 2 * Real.sin α * Real.cos α + 4 * (Real.cos α)^2 = 28/13) := by
  sorry

end trigonometric_identities_l544_54435


namespace quadratic_inequality_l544_54481

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set of f ≥ 0
def solution_set (a b c : ℝ) : Set ℝ := {x | x ≤ -3 ∨ x ≥ 4}

-- Define the theorem
theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, x ∈ solution_set a b c ↔ f a b c x ≥ 0) :
  a > 0 ∧ 
  (∀ x, cx^2 - bx + a < 0 ↔ x < -1/4 ∨ x > 1/3) :=
sorry

end quadratic_inequality_l544_54481


namespace danny_bottle_caps_l544_54458

/-- Proves that Danny found 1 more bottle cap than he threw away. -/
theorem danny_bottle_caps (found : ℕ) (thrown_away : ℕ) (current : ℕ)
  (h1 : found = 36)
  (h2 : thrown_away = 35)
  (h3 : current = 22)
  : found - thrown_away = 1 := by
  sorry

end danny_bottle_caps_l544_54458


namespace abc_inequality_l544_54409

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end abc_inequality_l544_54409


namespace system_of_equations_l544_54401

theorem system_of_equations (x y c d : ℝ) 
  (eq1 : 8 * x - 5 * y = c)
  (eq2 : 12 * y - 18 * x = d)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (d_nonzero : d ≠ 0) :
  c / d = -16 / 27 := by
sorry

end system_of_equations_l544_54401


namespace geometric_series_cube_sum_l544_54416

theorem geometric_series_cube_sum (a r : ℝ) (hr : -1 < r ∧ r < 1) :
  (a / (1 - r) = 2) →
  (a^2 / (1 - r^2) = 6) →
  (a^3 / (1 - r^3) = 96/7) :=
by sorry

end geometric_series_cube_sum_l544_54416


namespace max_valid_sequence_length_l544_54429

/-- A sequence of integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℤ) :=
  (∀ i, (a i) + (a (i+1)) + (a (i+2)) + (a (i+3)) + (a (i+4)) > 0) ∧
  (∀ i, (a i) + (a (i+1)) + (a (i+2)) + (a (i+3)) + (a (i+4)) + (a (i+5)) + (a (i+6)) < 0)

/-- The maximum length of a valid sequence is 10 -/
theorem max_valid_sequence_length :
  (∃ (a : ℕ → ℤ) (n : ℕ), n = 10 ∧ ValidSequence (λ i => if i < n then a i else 0)) ∧
  (∀ (a : ℕ → ℤ) (n : ℕ), n > 10 → ¬ValidSequence (λ i => if i < n then a i else 0)) :=
sorry

end max_valid_sequence_length_l544_54429


namespace det_4523_equals_2_l544_54490

def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem det_4523_equals_2 : det2x2 4 5 2 3 = 2 := by sorry

end det_4523_equals_2_l544_54490


namespace periodic_and_zeros_l544_54464

-- Define a periodic function
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x, f (x + T) = f x

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_and_zeros (f : ℝ → ℝ) (a : ℝ) :
  (a ≠ 0 ∧ ∀ x, f (x + a) = -f x) →
  IsPeriodic f (2 * a) ∧
  (IsOdd f → (∀ x, f (x + 1) = -f x) →
    ∃ (zeros : Finset ℝ), zeros.card ≥ 4035 ∧
      (∀ x ∈ zeros, -2017 ≤ x ∧ x ≤ 2017 ∧ f x = 0)) :=
by sorry


end periodic_and_zeros_l544_54464


namespace circle_tangency_line_intersection_l544_54441

-- Define the circle C
def circle_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Part I
theorem circle_tangency (m : ℝ) :
  (∃ x y : ℝ, circle_C m x y ∧ unit_circle x y) →
  (∀ x y : ℝ, circle_C m x y → ¬(unit_circle x y)) →
  m = 9 :=
sorry

-- Part II
theorem line_intersection (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C m x₁ y₁ ∧ circle_C m x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 14) →
  m = 10 :=
sorry

end circle_tangency_line_intersection_l544_54441


namespace robins_camera_pictures_l544_54425

theorem robins_camera_pictures :
  ∀ (phone_pics camera_pics total_pics albums pics_per_album : ℕ),
    phone_pics = 35 →
    albums = 5 →
    pics_per_album = 8 →
    total_pics = albums * pics_per_album →
    total_pics = phone_pics + camera_pics →
    camera_pics = 5 := by
  sorry

end robins_camera_pictures_l544_54425


namespace remainder_of_n_l544_54447

theorem remainder_of_n (n : ℕ) (h1 : n^3 % 7 = 3) (h2 : n^4 % 7 = 2) : n % 7 = 6 := by
  sorry

end remainder_of_n_l544_54447


namespace consecutive_integers_around_sqrt_seven_l544_54427

theorem consecutive_integers_around_sqrt_seven (a b : ℤ) : 
  a < Real.sqrt 7 ∧ Real.sqrt 7 < b ∧ b = a + 1 → a + b = 5 := by
  sorry

end consecutive_integers_around_sqrt_seven_l544_54427


namespace prob_more_ones_than_eights_l544_54439

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of rolling more 1's than 8's when rolling five fair eight-sided dice -/
def probMoreOnesThanEights : ℚ := 14026 / 32768

/-- Theorem stating that the probability of rolling more 1's than 8's is correct -/
theorem prob_more_ones_than_eights :
  let totalOutcomes : ℕ := numSides ^ numDice
  let probEqualOnesAndEights : ℚ := 4716 / totalOutcomes
  probMoreOnesThanEights = (1 - probEqualOnesAndEights) / 2 :=
sorry

end prob_more_ones_than_eights_l544_54439


namespace area_G1G2G3_l544_54411

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define point P inside triangle ABC
variable (P : ℝ × ℝ)

-- Define G1, G2, G3 as centroids of triangles PBC, PCA, PAB respectively
def G1 : ℝ × ℝ := sorry
def G2 : ℝ × ℝ := sorry
def G3 : ℝ × ℝ := sorry

-- Define the area function
def area (a b c : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_G1G2G3 (h : area A B C = 24) :
  area G1 G2 G3 = 8/3 := by sorry

end area_G1G2G3_l544_54411


namespace equation_solution_l544_54460

theorem equation_solution : ∃ b : ℝ, ∀ a : ℝ, (-6) * a^2 = 3 * (4*a + b) ∧ (a = 1 → b = -6) := by
  sorry

end equation_solution_l544_54460


namespace stratified_sampling_l544_54430

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the number of people sampled from each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- The stratified sampling theorem -/
theorem stratified_sampling
  (pop : Population)
  (sample : Sample)
  (h1 : pop.elderly = 27)
  (h2 : pop.middleAged = 54)
  (h3 : pop.young = 81)
  (h4 : sample.elderly = 6)
  (h5 : sample.middleAged / pop.middleAged = sample.elderly / pop.elderly)
  (h6 : sample.young / pop.young = sample.elderly / pop.elderly) :
  sample.elderly + sample.middleAged + sample.young = 36 :=
sorry

end stratified_sampling_l544_54430


namespace two_equidistant_points_l544_54437

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its normal vector and a point on the line -/
structure Line where
  normal : ℝ × ℝ
  point : ℝ × ℝ

/-- Configuration of a circle and two parallel tangent lines -/
structure CircleTangentConfig where
  circle : Circle
  tangent1 : Line
  tangent2 : Line
  d1 : ℝ  -- distance from circle center to tangent1
  d2 : ℝ  -- distance from circle center to tangent2

/-- Predicate to check if a point is equidistant from a circle and two lines -/
def isEquidistant (p : ℝ × ℝ) (c : Circle) (l1 l2 : Line) : Prop := sorry

/-- Main theorem: There are exactly two points equidistant from the circle and both tangents -/
theorem two_equidistant_points (config : CircleTangentConfig) 
  (h1 : config.d1 ≠ config.d2)
  (h2 : config.d1 > config.circle.radius)
  (h3 : config.d2 > config.circle.radius)
  (h4 : config.tangent1.normal = config.tangent2.normal) :  -- parallel tangents
  ∃! (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧ 
    isEquidistant p1 config.circle config.tangent1 config.tangent2 ∧ 
    isEquidistant p2 config.circle config.tangent1 config.tangent2 := by
  sorry

end two_equidistant_points_l544_54437


namespace chess_tournament_games_l544_54486

/-- The number of games played in a round-robin chess tournament. -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 21 participants, where each participant
    plays exactly one game with each of the remaining participants, 
    the total number of games played is 210. -/
theorem chess_tournament_games :
  num_games 21 = 210 := by
  sorry

end chess_tournament_games_l544_54486


namespace paving_job_units_l544_54448

theorem paving_job_units (worker1_rate worker2_rate reduced_efficiency total_time : ℝ) 
  (h1 : worker1_rate = 1 / 8)
  (h2 : worker2_rate = 1 / 12)
  (h3 : reduced_efficiency = 8)
  (h4 : total_time = 6) :
  let combined_rate := worker1_rate + worker2_rate - reduced_efficiency / total_time
  total_time * combined_rate = 192 := by
sorry

end paving_job_units_l544_54448


namespace cyclic_quadrilateral_angle_l544_54483

/-- In a cyclic quadrilateral ABCD where ∠A : ∠B : ∠C = 1 : 2 : 3, ∠D = 90° -/
theorem cyclic_quadrilateral_angle (A B C D : Real) (h1 : A + C = 180) (h2 : B + D = 180)
  (h3 : A / B = 1 / 2) (h4 : B / C = 2 / 3) : D = 90 := by
  sorry

end cyclic_quadrilateral_angle_l544_54483


namespace tony_age_in_six_years_l544_54476

/-- Given Jacob's current age and Tony's age relative to Jacob's, 
    calculate Tony's age after a certain number of years. -/
def tony_future_age (jacob_age : ℕ) (years_passed : ℕ) : ℕ :=
  (jacob_age / 2) + years_passed

/-- Theorem: Tony will be 18 years old in 6 years -/
theorem tony_age_in_six_years :
  tony_future_age 24 6 = 18 := by
  sorry

end tony_age_in_six_years_l544_54476


namespace congruence_mod_nine_l544_54495

theorem congruence_mod_nine : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2222 ≡ n [ZMOD 9] := by sorry

end congruence_mod_nine_l544_54495


namespace fourth_power_sum_of_cubic_roots_l544_54479

theorem fourth_power_sum_of_cubic_roots (a b c : ℝ) : 
  (a^3 - 3*a + 1 = 0) → 
  (b^3 - 3*b + 1 = 0) → 
  (c^3 - 3*c + 1 = 0) → 
  a^4 + b^4 + c^4 = 18 := by
  sorry

end fourth_power_sum_of_cubic_roots_l544_54479


namespace cube_volume_problem_l544_54452

theorem cube_volume_problem (a : ℝ) : 
  a > 0 →  -- Ensuring the side length is positive
  (a + 2) * a * (a - 2) = a^3 - 8 → 
  a^3 = 8 :=
by sorry

end cube_volume_problem_l544_54452


namespace symmetric_complex_product_l544_54487

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  (z₁.re = 2 ∧ z₁.im = 1) →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  z₁ * z₂ = -5 := by
  sorry

end symmetric_complex_product_l544_54487


namespace train_length_l544_54413

/-- Given a train crossing a bridge, calculate its length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * 1000 / 3600 →
  crossing_time = 30 →
  bridge_length = 235 →
  train_speed * crossing_time - bridge_length = 140 := by
  sorry

end train_length_l544_54413


namespace range_of_a_l544_54467

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x - 4 ≤ 0}

-- Define the theorem
theorem range_of_a (a : ℝ) : B a ⊆ A ↔ 0 ≤ a ∧ a < 3 :=
sorry

end range_of_a_l544_54467


namespace three_x_squared_y_squared_l544_54428

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 3*x^2*y^2 = 588 := by
  sorry

end three_x_squared_y_squared_l544_54428


namespace mushroom_count_l544_54475

theorem mushroom_count :
  ∀ n m : ℕ,
  n ≤ 70 →
  m = (52 * n) / 100 →
  ∃ x : ℕ,
  x ≤ 3 ∧
  2 * (m - x) = n - 3 →
  n = 25 :=
by sorry

end mushroom_count_l544_54475


namespace smallest_upper_bound_l544_54463

/-- The set of functions satisfying the given conditions -/
def S : Set (ℕ → ℝ) :=
  {f | f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n : ℝ) / (n + 1) * f (2 * n)}

/-- The smallest natural number M such that for any f ∈ S and any n ∈ ℕ, f(n) < M -/
def M : ℕ := 10

theorem smallest_upper_bound : 
  (∀ f ∈ S, ∀ n, f n < M) ∧ 
  (∀ m < M, ∃ f ∈ S, ∃ n, f n ≥ m) :=
sorry

end smallest_upper_bound_l544_54463


namespace polynomial_factor_sum_l544_54496

theorem polynomial_factor_sum (m n : ℚ) : 
  (∀ y, my^2 + n*y + 2 = (y + 1)*(y + 2)) → m + n = 4 := by
  sorry

end polynomial_factor_sum_l544_54496


namespace truck_mileage_l544_54492

/-- Given a truck that travels 240 miles on 5 gallons of gas, 
    prove that it can travel 336 miles on 7 gallons of gas. -/
theorem truck_mileage (miles_on_five : ℝ) (gallons_five : ℝ) (gallons_seven : ℝ) 
  (h1 : miles_on_five = 240)
  (h2 : gallons_five = 5)
  (h3 : gallons_seven = 7) :
  (miles_on_five / gallons_five) * gallons_seven = 336 := by
sorry

end truck_mileage_l544_54492


namespace output_for_15_l544_54474

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 25 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 38 := by sorry

end output_for_15_l544_54474


namespace r₂_lower_bound_two_is_greatest_lower_bound_l544_54421

/-- The function f(x) = x² - r₂x + r₃ -/
noncomputable def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂*x + r₃

/-- The sequence {gₙ} defined recursively -/
noncomputable def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The theorem stating the lower bound on |r₂| -/
theorem r₂_lower_bound (r₂ r₃ : ℝ) :
  (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) →
  (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i + 1) > g r₂ r₃ i) →
  (∀ M : ℝ, ∃ n : ℕ, |g r₂ r₃ n| > M) →
  |r₂| > 2 :=
sorry

/-- The theorem stating that 2 is the greatest lower bound -/
theorem two_is_greatest_lower_bound :
  ∀ ε > 0, ∃ r₂ r₃ : ℝ,
    (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) ∧
    (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i + 1) > g r₂ r₃ i) ∧
    (∀ M : ℝ, ∃ n : ℕ, |g r₂ r₃ n| > M) ∧
    |r₂| < 2 + ε :=
sorry

end r₂_lower_bound_two_is_greatest_lower_bound_l544_54421


namespace sqrt_product_equals_two_l544_54434

theorem sqrt_product_equals_two : Real.sqrt 20 * Real.sqrt (1/5) = 2 := by
  sorry

end sqrt_product_equals_two_l544_54434


namespace greatest_integer_radius_of_circle_exists_greatest_integer_radius_greatest_integer_radius_is_8_l544_54404

theorem greatest_integer_radius_of_circle (r : ℕ) : 
  (r : ℝ) * (r : ℝ) * Real.pi < 75 * Real.pi → r ≤ 8 :=
by sorry

theorem exists_greatest_integer_radius : 
  ∃ (r : ℕ), (r : ℝ) * (r : ℝ) * Real.pi < 75 * Real.pi ∧ 
  ∀ (s : ℕ), (s : ℝ) * (s : ℝ) * Real.pi < 75 * Real.pi → s ≤ r :=
by sorry

theorem greatest_integer_radius_is_8 : 
  ∃! (r : ℕ), (r : ℝ) * (r : ℝ) * Real.pi < 75 * Real.pi ∧ 
  ∀ (s : ℕ), (s : ℝ) * (s : ℝ) * Real.pi < 75 * Real.pi → s ≤ r ∧ r = 8 :=
by sorry

end greatest_integer_radius_of_circle_exists_greatest_integer_radius_greatest_integer_radius_is_8_l544_54404


namespace first_part_speed_l544_54453

/-- Represents a train journey with two parts -/
structure TrainJourney where
  x : ℝ  -- distance of first part in km
  V : ℝ  -- speed of first part in kmph

/-- Theorem stating the speed of the first part of the journey -/
theorem first_part_speed (j : TrainJourney) (h1 : j.x > 0) 
    (h2 : (j.x / j.V) + (2 * j.x / 20) = (3 * j.x) / 24) : j.V = 40 := by
  sorry

#check first_part_speed

end first_part_speed_l544_54453


namespace imaginary_part_of_z_l544_54415

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : (1 - 2*I)*z = 5*I) : 
  z.im = 1 := by sorry

end imaginary_part_of_z_l544_54415


namespace inequality_solution_set_l544_54440

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (x - 1) * (x + a) > 0 ↔
    (a < -1 ∧ (x < -a ∨ x > 1)) ∨
    (a = -1 ∧ x ≠ 1) ∨
    (a > -1 ∧ (x < -a ∨ x > 1))) :=
by sorry

end inequality_solution_set_l544_54440


namespace same_birthday_probability_l544_54454

/-- The number of days in a year -/
def daysInYear : ℕ := 365

/-- The probability of two classmates having their birthdays on the same day -/
def birthdayProbability : ℚ := 1 / daysInYear

theorem same_birthday_probability :
  birthdayProbability = 1 / daysInYear := by
  sorry

end same_birthday_probability_l544_54454


namespace coat_price_reduction_l544_54468

theorem coat_price_reduction (original_price reduction : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction = 250) :
  (reduction / original_price) * 100 = 50 := by
  sorry

end coat_price_reduction_l544_54468


namespace same_grade_percentage_l544_54414

theorem same_grade_percentage (total_students : ℕ) (same_grade_students : ℕ) : 
  total_students = 40 →
  same_grade_students = 17 →
  (same_grade_students : ℚ) / (total_students : ℚ) * 100 = 42.5 := by
  sorry

end same_grade_percentage_l544_54414


namespace students_in_band_or_sports_l544_54498

theorem students_in_band_or_sports
  (total : ℕ)
  (band : ℕ)
  (sports : ℕ)
  (both : ℕ)
  (h1 : total = 320)
  (h2 : band = 85)
  (h3 : sports = 200)
  (h4 : both = 60) :
  band + sports - both = 225 :=
by sorry

end students_in_band_or_sports_l544_54498


namespace complex_expression_equality_l544_54455

theorem complex_expression_equality : 
  ∀ (z₁ z₂ : ℂ), 
    z₁ = 2 - I → 
    z₂ = -I → 
    z₁ / z₂ + Complex.abs z₂ = 2 + 2*I := by
sorry

end complex_expression_equality_l544_54455


namespace min_people_with_hat_and_glove_l544_54469

theorem min_people_with_hat_and_glove (n : ℕ) (gloves hats both : ℕ) : 
  n > 0 → 
  gloves = (3 * n) / 8 → 
  hats = (5 * n) / 6 → 
  both ≥ gloves + hats - n → 
  both ≥ 5 := by
  sorry

end min_people_with_hat_and_glove_l544_54469


namespace marbles_lost_l544_54426

theorem marbles_lost (initial_marbles remaining_marbles : ℕ) 
  (h1 : initial_marbles = 16) 
  (h2 : remaining_marbles = 9) : 
  initial_marbles - remaining_marbles = 7 := by
  sorry

end marbles_lost_l544_54426


namespace tree_planting_optimization_l544_54471

/-- Tree planting activity optimization problem -/
theorem tree_planting_optimization (total_families : ℕ) 
  (silver_poplars : ℕ) (purple_plums : ℕ) 
  (silver_poplar_time : ℚ) (purple_plum_time : ℚ) :
  total_families = 65 →
  silver_poplars = 150 →
  purple_plums = 160 →
  silver_poplar_time = 2/5 →
  purple_plum_time = 3/5 →
  ∃ (group_a_families : ℕ) (duration : ℚ),
    group_a_families = 25 ∧
    duration = 12/5 ∧
    group_a_families ≤ total_families ∧
    (group_a_families : ℚ) * silver_poplars * silver_poplar_time = 
      (total_families - group_a_families : ℚ) * purple_plums * purple_plum_time ∧
    duration = (group_a_families : ℚ) * silver_poplars * silver_poplar_time / group_a_families ∧
    ∀ (other_group_a : ℕ) (other_duration : ℚ),
      other_group_a ≤ total_families →
      (other_group_a : ℚ) * silver_poplars * silver_poplar_time = 
        (total_families - other_group_a : ℚ) * purple_plums * purple_plum_time →
      other_duration = (other_group_a : ℚ) * silver_poplars * silver_poplar_time / other_group_a →
      duration ≤ other_duration :=
by
  sorry

end tree_planting_optimization_l544_54471


namespace square_value_l544_54482

theorem square_value (square : ℚ) : 44 * 25 = square * 100 → square = 11 := by
  sorry

end square_value_l544_54482


namespace inequality_proof_l544_54407

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b + 1/a > a + 1/b := by
  sorry

end inequality_proof_l544_54407


namespace intersecting_triangles_circumcircle_containment_l544_54443

/-- A triangle in a plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set (ℝ × ℝ) :=
  sorry

/-- Two triangles intersect if they have a common point -/
def intersect (t1 t2 : Triangle) : Prop :=
  sorry

/-- A point is inside or on a circle -/
def inside_or_on_circle (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  sorry

theorem intersecting_triangles_circumcircle_containment 
  (t1 t2 : Triangle) (h : intersect t1 t2) :
  ∃ (i : Fin 3), inside_or_on_circle (t1.vertices i) (circumcircle t2) ∨
                 inside_or_on_circle (t2.vertices i) (circumcircle t1) :=
sorry

end intersecting_triangles_circumcircle_containment_l544_54443


namespace x_equals_4n_l544_54465

/-- Given that x is 3 times larger than n, and 2n + 3 is some percentage of 25, prove that x = 4n -/
theorem x_equals_4n (n x : ℝ) (p : ℝ) 
  (h1 : x = n + 3 * n) 
  (h2 : 2 * n + 3 = p / 100 * 25) : 
  x = 4 * n := by
sorry

end x_equals_4n_l544_54465


namespace g_composition_equals_514_l544_54494

def g (x : ℝ) : ℝ := 7 * x + 3

theorem g_composition_equals_514 : g (g (g 1)) = 514 := by
  sorry

end g_composition_equals_514_l544_54494


namespace a_less_than_2_necessary_not_sufficient_l544_54493

theorem a_less_than_2_necessary_not_sufficient :
  (∀ a : ℝ, a^2 < 2*a → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 2*a) :=
by sorry

end a_less_than_2_necessary_not_sufficient_l544_54493


namespace exam_score_calculation_l544_54449

/-- Proves that the number of marks awarded for a correct answer is 4 in the given exam scenario -/
theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_score : ℕ) 
  (h1 : total_questions = 150)
  (h2 : correct_answers = 120)
  (h3 : total_score = 420)
  (h4 : ∀ x : ℕ, x * correct_answers - 2 * (total_questions - correct_answers) = total_score → x = 4) :
  ∃ x : ℕ, x * correct_answers - 2 * (total_questions - correct_answers) = total_score ∧ x = 4 :=
by sorry

end exam_score_calculation_l544_54449


namespace reading_time_calculation_l544_54424

theorem reading_time_calculation (total_time math_time spelling_time : ℕ) 
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18) :
  total_time - (math_time + spelling_time) = 27 := by
  sorry

end reading_time_calculation_l544_54424


namespace orange_juice_price_l544_54497

def initial_money : ℕ := 86
def bread_price : ℕ := 3
def bread_quantity : ℕ := 3
def juice_quantity : ℕ := 3
def money_left : ℕ := 59

theorem orange_juice_price :
  ∃ (juice_price : ℕ),
    initial_money - (bread_price * bread_quantity + juice_price * juice_quantity) = money_left ∧
    juice_price = 6 :=
by sorry

end orange_juice_price_l544_54497


namespace f_increasing_decreasing_l544_54472

-- Define the function f(x) = x^3 - x^2 - x
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

-- Theorem statement
theorem f_increasing_decreasing :
  (∀ x < -1/3, f' x > 0) ∧
  (∀ x ∈ Set.Ioo (-1/3) 1, f' x < 0) ∧
  (∀ x > 1, f' x > 0) ∧
  (f 1 = -1) ∧
  (f' 1 = 0) :=
sorry

end f_increasing_decreasing_l544_54472


namespace cubic_root_sum_l544_54461

theorem cubic_root_sum (p q r : ℝ) : 
  (3 * p^3 - 5 * p^2 + 12 * p - 7 = 0) →
  (3 * q^3 - 5 * q^2 + 12 * q - 7 = 0) →
  (3 * r^3 - 5 * r^2 + 12 * r - 7 = 0) →
  (p + q + r = 5/3) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = -35/3 := by
  sorry

end cubic_root_sum_l544_54461


namespace difference_in_circumferences_l544_54477

/-- The difference in circumferences of two concentric circles -/
theorem difference_in_circumferences 
  (inner_diameter : ℝ) 
  (track_width : ℝ) 
  (h1 : inner_diameter = 50) 
  (h2 : track_width = 15) : 
  (inner_diameter + 2 * track_width) * π - inner_diameter * π = 30 * π := by
  sorry

end difference_in_circumferences_l544_54477


namespace integer_multiplication_result_l544_54462

theorem integer_multiplication_result (x : ℤ) : 
  (10 * x = 64 ∨ 10 * x = 32 ∨ 10 * x = 12 ∨ 10 * x = 25 ∨ 10 * x = 30) → 10 * x = 30 := by
sorry

end integer_multiplication_result_l544_54462


namespace infinite_capacitor_chain_effective_capacitance_l544_54485

/-- Given an infinitely long chain of capacitors, each with capacitance C,
    the effective capacitance Ce between any two adjacent points
    is equal to ((1 + √3) * C) / 2. -/
theorem infinite_capacitor_chain_effective_capacitance (C : ℝ) (Ce : ℝ) 
  (h1 : C > 0) -- Capacitance is always positive
  (h2 : Ce = C + Ce / (2 + Ce / C)) -- Relationship derived from the infinite chain
  : Ce = ((1 + Real.sqrt 3) * C) / 2 := by
  sorry

end infinite_capacitor_chain_effective_capacitance_l544_54485


namespace distributions_without_zhoubi_l544_54478

/-- Represents the number of books -/
def num_books : ℕ := 4

/-- Represents the number of students -/
def num_students : ℕ := 3

/-- Represents the total number of distribution methods -/
def total_distributions : ℕ := 36

/-- Represents the number of distribution methods where student A receives "Zhoubi Suanjing" -/
def distributions_with_zhoubi : ℕ := 12

/-- Theorem stating the number of distribution methods where A does not receive "Zhoubi Suanjing" -/
theorem distributions_without_zhoubi :
  total_distributions - distributions_with_zhoubi = 24 :=
by sorry

end distributions_without_zhoubi_l544_54478


namespace tips_fraction_of_income_l544_54459

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- The fraction of income from tips for a waitress -/
def fractionFromTips (income : WaitressIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: If tips are 11/4 of salary, then 11/15 of income is from tips -/
theorem tips_fraction_of_income 
  (income : WaitressIncome) 
  (h : income.tips = (11 / 4) * income.salary) : 
  fractionFromTips income = 11 / 15 := by
  sorry

#check tips_fraction_of_income

end tips_fraction_of_income_l544_54459


namespace unique_hair_color_assignment_l544_54491

/-- Represents the three people in the problem -/
inductive Person : Type
  | Belokurov : Person
  | Chernov : Person
  | Ryzhov : Person

/-- Represents the three hair colors in the problem -/
inductive HairColor : Type
  | Blond : HairColor
  | Brunette : HairColor
  | RedHaired : HairColor

/-- Represents the assignment of hair colors to people -/
def hairColorAssignment : Person → HairColor
  | Person.Belokurov => HairColor.RedHaired
  | Person.Chernov => HairColor.Blond
  | Person.Ryzhov => HairColor.Brunette

/-- Condition: No person has a hair color matching their surname -/
def noMatchingSurname (assignment : Person → HairColor) : Prop :=
  assignment Person.Belokurov ≠ HairColor.Blond ∧
  assignment Person.Chernov ≠ HairColor.Brunette ∧
  assignment Person.Ryzhov ≠ HairColor.RedHaired

/-- Condition: The brunette is not Belokurov -/
def brunetteNotBelokurov (assignment : Person → HairColor) : Prop :=
  assignment Person.Belokurov ≠ HairColor.Brunette

/-- Condition: All three hair colors are represented -/
def allColorsRepresented (assignment : Person → HairColor) : Prop :=
  (∃ p, assignment p = HairColor.Blond) ∧
  (∃ p, assignment p = HairColor.Brunette) ∧
  (∃ p, assignment p = HairColor.RedHaired)

/-- Main theorem: The given hair color assignment is the only one satisfying all conditions -/
theorem unique_hair_color_assignment :
  ∀ (assignment : Person → HairColor),
    noMatchingSurname assignment ∧
    brunetteNotBelokurov assignment ∧
    allColorsRepresented assignment →
    assignment = hairColorAssignment :=
by sorry

end unique_hair_color_assignment_l544_54491


namespace square_sum_reciprocal_l544_54433

theorem square_sum_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a - 1/b - 1/(a+b) = 0) : (b/a + a/b)^2 = 5 := by
  sorry

end square_sum_reciprocal_l544_54433


namespace quadratic_solution_property_l544_54423

theorem quadratic_solution_property (a b : ℝ) : 
  (a * 1^2 + b * 1 + 1 = 0) → (3 - a - b = 4) := by
  sorry

end quadratic_solution_property_l544_54423


namespace expression_values_l544_54402

theorem expression_values (a b c d x y : ℝ) : 
  (a + b = 0) → 
  (c * d = 1) → 
  (x = 4 ∨ x = -4) → 
  (y = -6) → 
  ((2 * x - c * d + 4 * (a + b) - y^2 = -29 ∧ x = 4) ∨ 
   (2 * x - c * d + 4 * (a + b) - y^2 = -45 ∧ x = -4)) :=
by sorry

end expression_values_l544_54402


namespace quadratic_equations_solutions_l544_54422

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 4 ∧ x₁^2 - 6*x₁ + 8 = 0 ∧ x₂^2 - 6*x₂ + 8 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 4 + Real.sqrt 15 ∧ x₂ = 4 - Real.sqrt 15 ∧ x₁^2 - 8*x₁ + 1 = 0 ∧ x₂^2 - 8*x₂ + 1 = 0) :=
by sorry

end quadratic_equations_solutions_l544_54422


namespace sum_mod_nine_l544_54400

theorem sum_mod_nine : (9023 + 9024 + 9025 + 9026 + 9027) % 9 = 2 := by
  sorry

end sum_mod_nine_l544_54400


namespace triangle_area_in_rectangle_l544_54473

/-- Given a rectangle with dimensions 30 cm by 28 cm containing four congruent right-angled triangles,
    where the hypotenuse of each triangle forms part of the rectangle's perimeter,
    the total area of the four triangles is 56 cm². -/
theorem triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  a + 2 * b = 30 →
  2 * b = 28 →
  4 * (1/2 * a * b) = 56 :=
by sorry

end triangle_area_in_rectangle_l544_54473


namespace nell_gave_jeff_168_cards_l544_54403

/-- The number of cards Nell gave to Jeff -/
def cards_to_jeff (initial : ℕ) (to_john : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_john - remaining

/-- Proof that Nell gave 168 cards to Jeff -/
theorem nell_gave_jeff_168_cards :
  cards_to_jeff 573 195 210 = 168 := by
  sorry

end nell_gave_jeff_168_cards_l544_54403


namespace dishonest_dealer_profit_dishonest_dealer_profit_result_l544_54436

/-- Calculates the overall percent profit for a dishonest dealer selling two products. -/
theorem dishonest_dealer_profit (weight_A weight_B : ℝ) (cost_A cost_B : ℝ) 
  (discount_A discount_B : ℝ) (purchase_A purchase_B : ℝ) : ℝ :=
  let actual_weight_A := weight_A / 1000 * purchase_A
  let actual_weight_B := weight_B / 1000 * purchase_B
  let cost_price_A := actual_weight_A * cost_A
  let cost_price_B := actual_weight_B * cost_B
  let total_cost_price := cost_price_A + cost_price_B
  let selling_price_A := purchase_A * cost_A
  let selling_price_B := purchase_B * cost_B
  let discounted_price_A := selling_price_A * (1 - discount_A)
  let discounted_price_B := selling_price_B * (1 - discount_B)
  let total_selling_price := discounted_price_A + discounted_price_B
  let profit := total_selling_price - total_cost_price
  let percent_profit := profit / total_cost_price * 100
  percent_profit

/-- The overall percent profit is approximately 30.99% -/
theorem dishonest_dealer_profit_result : 
  ∃ ε > 0, |dishonest_dealer_profit 700 750 60 80 0.05 0.03 6 12 - 30.99| < ε :=
sorry

end dishonest_dealer_profit_dishonest_dealer_profit_result_l544_54436


namespace min_yellow_surface_fraction_l544_54431

-- Define the cube dimensions
def large_cube_edge : ℕ := 4
def small_cube_edge : ℕ := 1

-- Define the number of cubes
def total_cubes : ℕ := 64
def blue_cubes : ℕ := 48
def yellow_cubes : ℕ := 16

-- Define the surface area of the large cube
def large_cube_surface_area : ℕ := 6 * large_cube_edge * large_cube_edge

-- Define the minimum number of yellow cubes that must be on the surface
def min_yellow_surface_cubes : ℕ := yellow_cubes - 1

-- Theorem statement
theorem min_yellow_surface_fraction :
  (min_yellow_surface_cubes : ℚ) / large_cube_surface_area = 5 / 32 := by
  sorry

end min_yellow_surface_fraction_l544_54431


namespace combined_length_legs_arms_l544_54406

/-- Calculates the combined length of legs and arms for two people given their heights and body proportions -/
theorem combined_length_legs_arms 
  (aisha_height : ℝ) 
  (benjamin_height : ℝ) 
  (aisha_legs_ratio : ℝ) 
  (aisha_arms_ratio : ℝ) 
  (benjamin_legs_ratio : ℝ) 
  (benjamin_arms_ratio : ℝ) 
  (h1 : aisha_height = 174) 
  (h2 : benjamin_height = 190) 
  (h3 : aisha_legs_ratio = 1/3) 
  (h4 : aisha_arms_ratio = 1/6) 
  (h5 : benjamin_legs_ratio = 3/7) 
  (h6 : benjamin_arms_ratio = 1/4) : 
  (aisha_legs_ratio * aisha_height + aisha_arms_ratio * aisha_height + 
   benjamin_legs_ratio * benjamin_height + benjamin_arms_ratio * benjamin_height) = 215.93 := by
  sorry

end combined_length_legs_arms_l544_54406


namespace quadratic_roots_product_l544_54466

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 11 * p - 20 = 0) → 
  (3 * q^2 + 11 * q - 20 = 0) → 
  (5 * p - 4) * (3 * q - 2) = -89/3 := by
sorry

end quadratic_roots_product_l544_54466


namespace solution_set_part_i_range_of_a_part_ii_l544_54405

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + 1| - |x - a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 4 x > 2} = {x : ℝ | x < -7 ∨ x > 5/3} := by sorry

-- Part II
theorem range_of_a_part_ii :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 2 3, f a x ≥ |x - 4|) → a ∈ Set.Icc (-1) 5 := by sorry

end solution_set_part_i_range_of_a_part_ii_l544_54405


namespace fourth_root_sixteen_times_cube_root_eight_times_sqrt_four_l544_54418

theorem fourth_root_sixteen_times_cube_root_eight_times_sqrt_four : 
  (16 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (4 : ℝ) ^ (1/2) = 8 := by
  sorry

end fourth_root_sixteen_times_cube_root_eight_times_sqrt_four_l544_54418


namespace inequality_system_solution_l544_54446

/-- Given that the solution set of the inequality system {x + 1 > 2x - 2, x < a} is x < 3,
    prove that the range of values for a is a ≥ 3. -/
theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 1 > 2*x - 2 ∧ x < a) ↔ x < 3) → a ≥ 3 :=
by sorry

end inequality_system_solution_l544_54446
