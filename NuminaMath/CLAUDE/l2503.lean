import Mathlib

namespace cylinder_volume_with_inscribed_sphere_l2503_250397

/-- The volume of a cylinder with an inscribed sphere (tangent to top, bottom, and side) is 2π, 
    given that the volume of the inscribed sphere is 4π/3. -/
theorem cylinder_volume_with_inscribed_sphere (r : ℝ) (h : ℝ) :
  (4 / 3 * π * r^3 = 4 * π / 3) →
  (π * r^2 * h = 2 * π) :=
by sorry

end cylinder_volume_with_inscribed_sphere_l2503_250397


namespace problem_1_problem_2_l2503_250340

-- Problem 1
theorem problem_1 (a b : ℝ) : (a * b) ^ 6 / (a * b) ^ 2 * (a * b) ^ 4 = a ^ 8 * b ^ 8 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (3 * x ^ 3) ^ 2 * x ^ 5 - (-x ^ 2) ^ 6 / x = 8 * x ^ 11 := by
  sorry

end problem_1_problem_2_l2503_250340


namespace perpendicular_vectors_l2503_250335

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem perpendicular_vectors (x : ℝ) :
  (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) →
  x = 2 := by
  sorry

end perpendicular_vectors_l2503_250335


namespace article_gain_percentage_l2503_250323

/-- Calculates the percentage gain when selling an article -/
def percentageGain (costPrice sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Calculates the cost price given a selling price and loss percentage -/
def calculateCostPrice (sellingPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  sellingPrice / (1 - lossPercentage / 100)

theorem article_gain_percentage :
  let lossPrice : ℚ := 102
  let gainPrice : ℚ := 144
  let lossPercentage : ℚ := 15
  let costPrice := calculateCostPrice lossPrice lossPercentage
  percentageGain costPrice gainPrice = 20 := by sorry

end article_gain_percentage_l2503_250323


namespace total_cash_realized_l2503_250372

/-- Calculates the cash realized from selling a stock -/
def cashRealized (value : ℝ) (returnRate : ℝ) (brokerageFeeRate : ℝ) : ℝ :=
  let grossValue := value * (1 + returnRate)
  grossValue * (1 - brokerageFeeRate)

/-- Theorem: The total cash realized from selling all three stocks is $65,120.75 -/
theorem total_cash_realized :
  let stockA := cashRealized 10000 0.14 0.0025
  let stockB := cashRealized 20000 0.10 0.005
  let stockC := cashRealized 30000 0.07 0.0075
  stockA + stockB + stockC = 65120.75 := by
  sorry

end total_cash_realized_l2503_250372


namespace sum_of_x_and_y_on_circle_l2503_250333

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 48) : x + y = 2 := by
  sorry

end sum_of_x_and_y_on_circle_l2503_250333


namespace sum_of_fourth_powers_l2503_250396

theorem sum_of_fourth_powers (a b : ℝ) 
  (h1 : a^2 - b^2 = 8) 
  (h2 : a * b = 2) : 
  a^4 + b^4 = 72 := by
sorry

end sum_of_fourth_powers_l2503_250396


namespace fraction_value_l2503_250316

theorem fraction_value (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 - 2*x + 1) / (x^2 - 1) = 1 - Real.sqrt 2 := by
  sorry

end fraction_value_l2503_250316


namespace tom_spent_seven_tickets_on_hat_l2503_250373

/-- The number of tickets Tom spent on the hat -/
def tickets_spent_on_hat (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (tickets_left : ℕ) : ℕ :=
  whack_a_mole_tickets + skee_ball_tickets - tickets_left

/-- Theorem stating that Tom spent 7 tickets on the hat -/
theorem tom_spent_seven_tickets_on_hat :
  tickets_spent_on_hat 32 25 50 = 7 := by
  sorry

end tom_spent_seven_tickets_on_hat_l2503_250373


namespace line_intersects_circle_midpoint_trajectory_line_equations_with_ratio_l2503_250313

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line L
def line_L (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the fixed point P
def point_P : ℝ × ℝ := (1, 1)

-- Theorem 1: Line L always intersects circle C at two distinct points
theorem line_intersects_circle (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ :=
sorry

-- Theorem 2: Trajectory of midpoint M
theorem midpoint_trajectory (x y : ℝ) :
  (∃ (m : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ ∧
    x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2) ↔
  x^2 + y^2 - x - 2*y + 1 = 0 :=
sorry

-- Theorem 3: Equations of line L when P divides AB in 1:2 ratio
theorem line_equations_with_ratio :
  ∃ (m : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ ∧
    2 * (point_P.1 - x₁) = x₂ - point_P.1 ∧
    2 * (point_P.2 - y₁) = y₂ - point_P.2 ↔
  (∀ x y, x - y = 0 ∨ x + y - 2 = 0) :=
sorry

end line_intersects_circle_midpoint_trajectory_line_equations_with_ratio_l2503_250313


namespace four_possible_d_values_l2503_250357

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition of two 5-digit numbers resulting in another 5-digit number -/
def ValidAddition (a b c d : Digit) : Prop :=
  ∃ (n : ℕ), n < 100000 ∧
  10000 * a.val + 1000 * b.val + 100 * c.val + 10 * d.val + a.val +
  10000 * c.val + 1000 * b.val + 100 * a.val + 10 * d.val + d.val =
  10000 * d.val + 1000 * d.val + 100 * d.val + 10 * c.val + b.val

/-- The main theorem stating that there are exactly 4 possible values for D -/
theorem four_possible_d_values :
  ∃! (s : Finset Digit), s.card = 4 ∧
  ∀ d : Digit, d ∈ s ↔
    ∃ (a b c : Digit), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧ a ≠ c ∧ b ≠ d ∧
    ValidAddition a b c d :=
sorry

end four_possible_d_values_l2503_250357


namespace sequence_fourth_term_l2503_250388

/-- Given a sequence a_n where a_1 = 2 and {1+a_n} is a geometric sequence
    with common ratio 3, prove that a_4 = 80 -/
theorem sequence_fourth_term (a : ℕ → ℝ) : 
  a 1 = 2 ∧ 
  (∀ n : ℕ, (1 + a (n + 1)) = 3 * (1 + a n)) →
  a 4 = 80 := by
sorry

end sequence_fourth_term_l2503_250388


namespace cuboid_sphere_surface_area_l2503_250350

-- Define the cuboid
structure Cuboid where
  face_area1 : ℝ
  face_area2 : ℝ
  face_area3 : ℝ
  vertices_on_sphere : Bool

-- Define the theorem
theorem cuboid_sphere_surface_area 
  (c : Cuboid) 
  (h1 : c.face_area1 = 12) 
  (h2 : c.face_area2 = 15) 
  (h3 : c.face_area3 = 20) 
  (h4 : c.vertices_on_sphere = true) : 
  ∃ (sphere_surface_area : ℝ), sphere_surface_area = 50 * Real.pi :=
sorry

end cuboid_sphere_surface_area_l2503_250350


namespace tan_a_value_l2503_250392

theorem tan_a_value (a : Real) (h : Real.tan (a + π/4) = 1/7) : 
  Real.tan a = -3/4 := by
  sorry

end tan_a_value_l2503_250392


namespace log_equation_holds_l2503_250346

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by sorry

end log_equation_holds_l2503_250346


namespace sqrt_D_irrational_l2503_250318

theorem sqrt_D_irrational (a : ℤ) : 
  let b : ℤ := a + 2
  let c : ℤ := a^2 + b
  let D : ℤ := a^2 + b^2 + c^2
  Irrational (Real.sqrt D) := by
sorry

end sqrt_D_irrational_l2503_250318


namespace crease_length_in_folded_rectangle_l2503_250327

/-- Represents a folded rectangle with given dimensions and fold properties -/
structure FoldedRectangle where
  width : ℝ
  fold_distance : ℝ
  crease_length : ℝ
  fold_angle : ℝ

/-- Theorem stating the crease length in a specific folded rectangle configuration -/
theorem crease_length_in_folded_rectangle (r : FoldedRectangle) 
  (h1 : r.width = 8)
  (h2 : r.fold_distance = 2)
  (h3 : Real.tan r.fold_angle = 3) : 
  r.crease_length = 2/3 := by
  sorry

#check crease_length_in_folded_rectangle

end crease_length_in_folded_rectangle_l2503_250327


namespace rhombus_area_l2503_250337

/-- Theorem: Area of a rhombus with given side and diagonal lengths -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (area : ℝ) : 
  side = 26 → diagonal1 = 20 → area = 480 → 
  ∃ (diagonal2 : ℝ), 
    diagonal2 ^ 2 = 4 * (side ^ 2 - (diagonal1 / 2) ^ 2) ∧ 
    area = (diagonal1 * diagonal2) / 2 := by
  sorry


end rhombus_area_l2503_250337


namespace stadium_seats_problem_l2503_250345

/-- Represents the number of seats in the n-th row -/
def a (n : ℕ) : ℕ := n + 1

/-- The total number of seats in the first n rows -/
def total_seats (n : ℕ) : ℕ := n * (n + 3) / 2

/-- The sum of the first n terms of the sequence a_n / (n(n+1)^2) -/
def S (n : ℕ) : ℚ := n / (n + 1)

theorem stadium_seats_problem :
  (total_seats 20 = 230) ∧ (S 20 = 20 / 21) := by
  sorry

end stadium_seats_problem_l2503_250345


namespace line_in_quadrants_l2503_250355

-- Define a line y = kx + b
structure Line where
  k : ℝ
  b : ℝ

-- Define quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to check if a line passes through a quadrant
def passesThrough (l : Line) (q : Quadrant) : Prop := sorry

-- Theorem statement
theorem line_in_quadrants (l : Line) :
  passesThrough l Quadrant.first ∧ 
  passesThrough l Quadrant.third ∧ 
  passesThrough l Quadrant.fourth →
  l.k > 0 := by sorry

end line_in_quadrants_l2503_250355


namespace don_earlier_rum_amount_l2503_250389

/-- The amount of rum Don had earlier in the day -/
def donEarlierRum (pancakeRum : ℝ) (maxMultiplier : ℝ) (remainingRum : ℝ) : ℝ :=
  maxMultiplier * pancakeRum - (pancakeRum + remainingRum)

/-- Theorem stating that Don had 12oz of rum earlier that day -/
theorem don_earlier_rum_amount :
  donEarlierRum 10 3 8 = 12 := by
  sorry

end don_earlier_rum_amount_l2503_250389


namespace expression_approximation_l2503_250317

theorem expression_approximation : 
  let x := Real.sqrt 1.1 / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt 0.49
  ∃ ε > 0, ε < 0.00005 ∧ |x - 2.8793| < ε :=
by sorry

end expression_approximation_l2503_250317


namespace elevator_problem_l2503_250349

theorem elevator_problem :
  let num_elevators : ℕ := 4
  let num_people : ℕ := 3
  let num_same_elevator : ℕ := 2
  (Nat.choose num_people num_same_elevator) * (num_elevators * (num_elevators - 1)) = 36
  := by sorry

end elevator_problem_l2503_250349


namespace completing_square_result_l2503_250307

/-- Represents the completing the square method applied to a quadratic equation -/
def completing_square (a b c : ℝ) : ℝ × ℝ := sorry

theorem completing_square_result :
  let (p, q) := completing_square 1 4 3
  p = 2 ∧ q = 1 := by sorry

end completing_square_result_l2503_250307


namespace add_1850_minutes_to_3_15pm_l2503_250326

/-- Represents a time of day in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Adds minutes to a given time and wraps around to the next day if necessary -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

/-- Converts a number of minutes to days, hours, and minutes -/
def minutesToDHM (m : Nat) : (Nat × Nat × Nat) :=
  sorry

theorem add_1850_minutes_to_3_15pm (start : Time) (h : start.hours = 15 ∧ start.minutes = 15) :
  let end_time := addMinutes start 1850
  end_time.hours = 22 ∧ end_time.minutes = 5 ∧ (minutesToDHM 1850).1 = 1 := by
  sorry

end add_1850_minutes_to_3_15pm_l2503_250326


namespace max_gemstone_value_is_72_l2503_250312

/-- Represents a type of gemstone with its weight and value --/
structure Gemstone where
  weight : ℕ
  value : ℕ

/-- The problem setup --/
def treasureHuntProblem :=
  let sapphire : Gemstone := ⟨6, 15⟩
  let ruby : Gemstone := ⟨3, 9⟩
  let diamond : Gemstone := ⟨2, 5⟩
  let maxWeight : ℕ := 24
  let minEachType : ℕ := 10
  (sapphire, ruby, diamond, maxWeight, minEachType)

/-- The maximum value of gemstones that can be carried --/
def maxGemstoneValue (problem : Gemstone × Gemstone × Gemstone × ℕ × ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum value is 72 --/
theorem max_gemstone_value_is_72 :
  maxGemstoneValue treasureHuntProblem = 72 := by
  sorry

end max_gemstone_value_is_72_l2503_250312


namespace max_value_x2_plus_y2_l2503_250334

theorem max_value_x2_plus_y2 (x y : ℝ) (h : 3 * x^2 + 2 * y^2 = 2 * x) :
  ∃ (M : ℝ), M = 4/9 ∧ x^2 + y^2 ≤ M ∧ ∃ (x₀ y₀ : ℝ), 3 * x₀^2 + 2 * y₀^2 = 2 * x₀ ∧ x₀^2 + y₀^2 = M :=
sorry

end max_value_x2_plus_y2_l2503_250334


namespace upstream_speed_calculation_l2503_250309

/-- Represents the speed of a man rowing in different water conditions -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the speed of the man rowing upstream given his rowing speeds in still water and downstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the man's speed in still water is 20 kmph and downstream is 33 kmph, his upstream speed is 7 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 20) 
  (h2 : s.downstream = 33) : 
  upstreamSpeed s = 7 := by
  sorry

#eval upstreamSpeed { stillWater := 20, downstream := 33 }

end upstream_speed_calculation_l2503_250309


namespace people_off_first_stop_l2503_250315

/-- Represents the number of people who got off at the first stop -/
def first_stop_off : ℕ := sorry

/-- The initial number of people on the bus -/
def initial_people : ℕ := 50

/-- The number of people who got off at the second stop -/
def second_stop_off : ℕ := 8

/-- The number of people who got on at the second stop -/
def second_stop_on : ℕ := 2

/-- The number of people who got off at the third stop -/
def third_stop_off : ℕ := 4

/-- The number of people who got on at the third stop -/
def third_stop_on : ℕ := 3

/-- The final number of people on the bus after the third stop -/
def final_people : ℕ := 28

theorem people_off_first_stop :
  initial_people - first_stop_off - (second_stop_off - second_stop_on) - (third_stop_off - third_stop_on) = final_people ∧
  first_stop_off = 15 := by sorry

end people_off_first_stop_l2503_250315


namespace ellipse_midpoint_property_l2503_250366

noncomputable section

-- Define the ellipse C
def C : Set (ℝ × ℝ) := {p | p.1^2 / 3 + p.2^2 = 1}

-- Define vertices A₁ and A₂
def A₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
def A₂ : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the line x = -2√3
def line_P : Set (ℝ × ℝ) := {p | p.1 = -2 * Real.sqrt 3}

-- Main theorem
theorem ellipse_midpoint_property 
  (P : ℝ × ℝ) 
  (h_P : P ∈ line_P ∧ P.2 ≠ 0) 
  (M N : ℝ × ℝ) 
  (h_M : M ∈ C ∧ ∃ t : ℝ, M = (1 - t) • P + t • A₂) 
  (h_N : N ∈ C ∧ ∃ s : ℝ, N = (1 - s) • P + s • A₁) 
  (Q : ℝ × ℝ) 
  (h_Q : Q = (M + N) / 2) : 
  2 * dist A₁ Q = dist M N :=
sorry

end

end ellipse_midpoint_property_l2503_250366


namespace complex_addition_subtraction_l2503_250391

theorem complex_addition_subtraction : 
  (1 : ℂ) * (5 - 6 * I) + (-2 - 2 * I) - (3 + 3 * I) = -11 * I := by sorry

end complex_addition_subtraction_l2503_250391


namespace deductive_reasoning_form_not_sufficient_l2503_250314

/-- A structure representing a deductive argument --/
structure DeductiveArgument where
  premises : List Prop
  conclusion : Prop
  form_correct : Bool

/-- A predicate that determines if a deductive argument is valid --/
def is_valid (arg : DeductiveArgument) : Prop :=
  arg.form_correct ∧ (∀ p ∈ arg.premises, p) → arg.conclusion

/-- Theorem stating that conforming to the form of deductive reasoning alone
    is not sufficient to guarantee the correctness of the conclusion --/
theorem deductive_reasoning_form_not_sufficient :
  ∃ (arg : DeductiveArgument), arg.form_correct ∧ ¬arg.conclusion :=
sorry

end deductive_reasoning_form_not_sufficient_l2503_250314


namespace guitar_purchase_savings_l2503_250378

/-- Proves that the difference in cost between Guitar Center and Sweetwater is $50 --/
theorem guitar_purchase_savings (retail_price : ℝ) 
  (gc_discount_rate : ℝ) (gc_shipping_fee : ℝ) 
  (sw_discount_rate : ℝ) :
  retail_price = 1000 →
  gc_discount_rate = 0.15 →
  gc_shipping_fee = 100 →
  sw_discount_rate = 0.10 →
  (retail_price * (1 - gc_discount_rate) + gc_shipping_fee) -
  (retail_price * (1 - sw_discount_rate)) = 50 := by
sorry

end guitar_purchase_savings_l2503_250378


namespace probability_at_least_one_cherry_plum_probability_at_least_one_cherry_plum_proof_l2503_250308

/-- The probability of selecting at least one cherry plum cutting -/
theorem probability_at_least_one_cherry_plum 
  (total_cuttings : ℕ) 
  (cherry_plum_cuttings : ℕ) 
  (plum_cuttings : ℕ) 
  (selected_cuttings : ℕ)
  (h1 : total_cuttings = 20)
  (h2 : cherry_plum_cuttings = 8)
  (h3 : plum_cuttings = 12)
  (h4 : selected_cuttings = 3)
  (h5 : total_cuttings = cherry_plum_cuttings + plum_cuttings) : 
  ℚ :=
46/57

theorem probability_at_least_one_cherry_plum_proof 
  (total_cuttings : ℕ) 
  (cherry_plum_cuttings : ℕ) 
  (plum_cuttings : ℕ) 
  (selected_cuttings : ℕ)
  (h1 : total_cuttings = 20)
  (h2 : cherry_plum_cuttings = 8)
  (h3 : plum_cuttings = 12)
  (h4 : selected_cuttings = 3)
  (h5 : total_cuttings = cherry_plum_cuttings + plum_cuttings) : 
  probability_at_least_one_cherry_plum total_cuttings cherry_plum_cuttings plum_cuttings selected_cuttings h1 h2 h3 h4 h5 = 46/57 := by
  sorry

end probability_at_least_one_cherry_plum_probability_at_least_one_cherry_plum_proof_l2503_250308


namespace gcd_consecutive_fib_46368_75025_l2503_250370

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem: The GCD of two consecutive Fibonacci numbers 46368 and 75025 is 1 -/
theorem gcd_consecutive_fib_46368_75025 :
  ∃ n : ℕ, fib n = 46368 ∧ fib (n + 1) = 75025 ∧ Nat.gcd 46368 75025 = 1 := by
  sorry

end gcd_consecutive_fib_46368_75025_l2503_250370


namespace semicircle_radius_in_isosceles_triangle_exists_isosceles_triangle_with_inscribed_semicircle_l2503_250352

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithInscribedSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ

/-- Theorem stating the relationship between the triangle's dimensions and the semicircle's radius -/
theorem semicircle_radius_in_isosceles_triangle 
  (triangle : IsoscelesTriangleWithInscribedSemicircle) 
  (h_base : triangle.base = 20) 
  (h_height : triangle.height = 18) : 
  triangle.radius = 90 / Real.sqrt 106 := by
  sorry

/-- Existence of the isosceles triangle with the given properties -/
theorem exists_isosceles_triangle_with_inscribed_semicircle :
  ∃ (triangle : IsoscelesTriangleWithInscribedSemicircle), 
    triangle.base = 20 ∧ 
    triangle.height = 18 ∧ 
    triangle.radius = 90 / Real.sqrt 106 := by
  sorry

end semicircle_radius_in_isosceles_triangle_exists_isosceles_triangle_with_inscribed_semicircle_l2503_250352


namespace function_relationship_l2503_250377

theorem function_relationship (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  ¬(∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3 →
    ∀ x y : ℝ, x < y → a^x > a^y) :=
by sorry

end function_relationship_l2503_250377


namespace gcd_12a_18b_min_l2503_250375

theorem gcd_12a_18b_min (a b : ℕ+) (h : Nat.gcd a b = 9) :
  (∃ (a' b' : ℕ+), Nat.gcd a' b' = 9 ∧ Nat.gcd (12 * a') (18 * b') = 54) ∧
  (Nat.gcd (12 * a) (18 * b) ≥ 54) :=
sorry

end gcd_12a_18b_min_l2503_250375


namespace isosceles_right_triangle_area_l2503_250384

/-- Given an isosceles right triangle with perimeter 2p, its area is (3-2√2)p² -/
theorem isosceles_right_triangle_area (p : ℝ) (h : p > 0) : 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    (2 * x + x * Real.sqrt 2 = 2 * p) ∧ 
    ((1 / 2) * x * x = (3 - 2 * Real.sqrt 2) * p^2) := by
  sorry

end isosceles_right_triangle_area_l2503_250384


namespace tan_five_pi_over_four_l2503_250301

theorem tan_five_pi_over_four : Real.tan (5 * π / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l2503_250301


namespace product_statistics_l2503_250305

def product_ratings : List ℝ := [9.6, 10.1, 9.7, 9.8, 10.0, 9.7, 10.0, 9.8, 10.1, 10.2]

def sum_of_squares : ℝ := 98.048

def improvement : ℝ := 0.2

def is_first_class (rating : ℝ) : Prop := rating ≥ 10

theorem product_statistics :
  let n : ℕ := product_ratings.length
  let mean : ℝ := (product_ratings.sum) / n
  let variance : ℝ := sum_of_squares / n - mean ^ 2
  let new_mean : ℝ := mean + improvement
  let new_variance : ℝ := variance
  (mean = 9.9) ∧
  (variance = 0.038) ∧
  (new_mean = 10.1) ∧
  (new_variance = 0.038) :=
sorry

end product_statistics_l2503_250305


namespace belinda_age_l2503_250356

theorem belinda_age (tony_age belinda_age : ℕ) : 
  tony_age + belinda_age = 56 →
  belinda_age = 2 * tony_age + 8 →
  tony_age = 16 →
  belinda_age = 40 := by
sorry

end belinda_age_l2503_250356


namespace parallelogram_height_l2503_250336

/-- Given a parallelogram with area 180 square centimeters and base 18 cm, its height is 10 cm. -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 180 → base = 18 → area = base * height → height = 10 := by
  sorry

end parallelogram_height_l2503_250336


namespace amit_left_after_three_days_l2503_250331

/-- The number of days Amit takes to complete the work alone -/
def amit_days : ℕ := 15

/-- The number of days Ananthu takes to complete the work alone -/
def ananthu_days : ℕ := 45

/-- The total number of days taken to complete the work -/
def total_days : ℕ := 39

/-- The number of days Amit worked before leaving -/
def amit_worked_days : ℕ := 3

theorem amit_left_after_three_days :
  ∃ (w : ℝ), w > 0 ∧
  amit_worked_days * (w / amit_days) + (total_days - amit_worked_days) * (w / ananthu_days) = w :=
sorry

end amit_left_after_three_days_l2503_250331


namespace probability_correct_l2503_250374

-- Define the number of red and blue marbles
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 9

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + blue_marbles

-- Define the number of marbles to be selected
def selected_marbles : ℕ := 4

-- Define the probability of selecting 2 red and 2 blue marbles
def probability_two_red_two_blue : ℚ := 4 / 27

-- Theorem statement
theorem probability_correct : 
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles selected_marbles = probability_two_red_two_blue := by
  sorry

end probability_correct_l2503_250374


namespace quadratic_minimum_ratio_bound_l2503_250371

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the derivative of the quadratic function
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

-- Define the second derivative of the quadratic function
def quadratic_second_derivative (a : ℝ) : ℝ := 2 * a

theorem quadratic_minimum_ratio_bound (a b c : ℝ) :
  a > 0 →  -- Ensures the function is concave up
  quadratic_derivative a b 0 > 0 →  -- f'(0) > 0
  (∀ x : ℝ, quadratic a b c x ≥ 0) →  -- f(x) ≥ 0 for all real x
  (quadratic a b c 1) / (quadratic_second_derivative a) ≥ 2 :=
by sorry

end quadratic_minimum_ratio_bound_l2503_250371


namespace absolute_value_of_two_is_not_negative_two_l2503_250385

theorem absolute_value_of_two_is_not_negative_two : ¬(|2| = -2) := by
  sorry

end absolute_value_of_two_is_not_negative_two_l2503_250385


namespace line_on_plane_perp_other_plane_implies_planes_perp_l2503_250367

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- The line lies on the plane -/
def lies_on (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- The line is perpendicular to the plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two planes are perpendicular -/
def perpendicular_planes (p1 p2 : Plane3D) : Prop :=
  sorry

theorem line_on_plane_perp_other_plane_implies_planes_perp
  (l : Line3D) (α β : Plane3D) :
  lies_on l α → perpendicular_line_plane l β → perpendicular_planes α β :=
by sorry

end line_on_plane_perp_other_plane_implies_planes_perp_l2503_250367


namespace equation_solutions_l2503_250319

theorem equation_solutions :
  (∃ x : ℚ, (x + 1) / 2 - (x - 3) / 6 = (5 * x + 1) / 3 + 1 ∧ x = -1/4) ∧
  (∃ x : ℚ, (x - 4) / (1/5) - 1 = (x - 3) / (1/2) ∧ x = 5) := by
  sorry

end equation_solutions_l2503_250319


namespace calcium_atomic_weight_l2503_250394

/-- The atomic weight of Oxygen -/
def atomic_weight_O : ℝ := 16

/-- The molecular weight of Calcium Oxide (CaO) -/
def molecular_weight_CaO : ℝ := 56

/-- The atomic weight of Calcium -/
def atomic_weight_Ca : ℝ := molecular_weight_CaO - atomic_weight_O

/-- Theorem stating that the atomic weight of Calcium is 40 -/
theorem calcium_atomic_weight :
  atomic_weight_Ca = 40 := by sorry

end calcium_atomic_weight_l2503_250394


namespace perpendicular_vectors_magnitude_l2503_250364

/-- Given vectors a and b in ℝ², if a ⊥ b, then |a| = 2 -/
theorem perpendicular_vectors_magnitude (x : ℝ) :
  let a : ℝ × ℝ := (x, Real.sqrt 3)
  let b : ℝ × ℝ := (3, -Real.sqrt 3)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⊥ b condition
  Real.sqrt (a.1^2 + a.2^2) = 2 := by
sorry

end perpendicular_vectors_magnitude_l2503_250364


namespace garbage_collection_average_l2503_250361

theorem garbage_collection_average (total_garbage : ℝ) 
  (h1 : total_garbage = 900) 
  (h2 : ∃ x : ℝ, total_garbage = x + x / 2) : 
  ∃ x : ℝ, x + x / 2 = total_garbage ∧ x = 600 :=
by sorry

end garbage_collection_average_l2503_250361


namespace additional_distance_for_average_speed_l2503_250322

def initial_distance : ℝ := 20
def initial_speed : ℝ := 25
def second_speed : ℝ := 40
def desired_average_speed : ℝ := 35

theorem additional_distance_for_average_speed :
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = desired_average_speed ∧
    additional_distance = 64 := by
  sorry

end additional_distance_for_average_speed_l2503_250322


namespace simple_interest_rate_percent_l2503_250344

/-- Given an amount of simple interest, time period, and rate, prove that the rate percent is correct. -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (rate : ℝ) 
  (h1 : interest = 400) 
  (h2 : time = 4) 
  (h3 : rate = 0.1) : 
  rate * 100 = 10 := by
sorry


end simple_interest_rate_percent_l2503_250344


namespace minimum_fuse_length_l2503_250339

theorem minimum_fuse_length (safe_distance : ℝ) (personnel_speed : ℝ) (fuse_burn_speed : ℝ) :
  safe_distance = 70 →
  personnel_speed = 7 →
  fuse_burn_speed = 10.3 →
  ∃ x : ℝ, x > 103 ∧ x / fuse_burn_speed > safe_distance / personnel_speed :=
by
  sorry

end minimum_fuse_length_l2503_250339


namespace base_85_congruence_l2503_250311

theorem base_85_congruence (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20) 
  (h3 : (74639281 : ℤ) - b ≡ 0 [ZMOD 17]) : b = 1 := by
  sorry

end base_85_congruence_l2503_250311


namespace reading_time_reduction_xiao_yu_reading_time_l2503_250369

/-- Represents the number of days to read a book given the pages per day -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- The theorem stating the relationship between reading rates and days to finish the book -/
theorem reading_time_reduction (initial_pages_per_day : ℕ) (initial_days : ℕ) (additional_pages : ℕ) :
  initial_pages_per_day > 0 →
  initial_days > 0 →
  additional_pages > 0 →
  days_to_read (initial_pages_per_day * initial_days) (initial_pages_per_day + additional_pages) =
    initial_days * initial_pages_per_day / (initial_pages_per_day + additional_pages) :=
by
  sorry

/-- The specific instance of the theorem for the given problem -/
theorem xiao_yu_reading_time :
  days_to_read (15 * 24) (15 + 3) = 20 :=
by
  sorry

end reading_time_reduction_xiao_yu_reading_time_l2503_250369


namespace friends_total_skittles_l2503_250398

/-- Given a person who gives a certain number of Skittles to each of their friends,
    calculate the total number of Skittles their friends have. -/
def total_skittles (skittles_per_friend : ℝ) (num_friends : ℝ) : ℝ :=
  skittles_per_friend * num_friends

/-- Theorem stating that if a person gives 40.0 Skittles to each of their 5.0 friends,
    the total number of Skittles their friends have is 200.0. -/
theorem friends_total_skittles :
  total_skittles 40.0 5.0 = 200.0 := by
  sorry

end friends_total_skittles_l2503_250398


namespace locus_P_is_correct_l2503_250320

/-- The locus of points P that are the second intersection of line OM and circle OAN,
    where O is the center of a circle with radius r, A(c, 0) is a point on its diameter,
    and M and N are symmetrical points on the circle with respect to OA. -/
def locus_P (r c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; (x^2 + y^2 - 2*c*x)^2 - r^2*(x^2 + y^2) = 0}

/-- Theorem stating that the locus_P is the correct description of the geometric locus. -/
theorem locus_P_is_correct (r c : ℝ) (hr : r > 0) (hc : c ≠ 0) :
  ∀ p : ℝ × ℝ, p ∈ locus_P r c ↔ 
    ∃ (m n : ℝ × ℝ),
      (∀ x y, (x, y) = m → x^2 + y^2 = r^2) ∧
      (∀ x y, (x, y) = n → x^2 + y^2 = r^2) ∧
      (∃ t, m.1 = t * n.1 ∧ m.2 = -t * n.2) ∧
      (∃ s, p = (s * m.1, s * m.2)) ∧
      (∃ u v, p.1^2 + p.2^2 + 2*u*p.1 + 2*v*p.2 = 0 ∧
              c^2 + 2*u*c = 0 ∧
              0^2 + 0^2 + 2*u*0 + 2*v*0 = 0) :=
by sorry

end locus_P_is_correct_l2503_250320


namespace rotated_angle_measure_l2503_250381

/-- Given an angle of 60 degrees rotated 540 degrees clockwise, 
    the resulting new acute angle is also 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 → 
  rotation = 540 → 
  (rotation % 360 - initial_angle) % 180 = 60 := by
sorry

end rotated_angle_measure_l2503_250381


namespace pencil_division_l2503_250341

theorem pencil_division (num_students num_pencils : ℕ) 
  (h1 : num_students = 2) 
  (h2 : num_pencils = 18) : 
  num_pencils / num_students = 9 := by
sorry

end pencil_division_l2503_250341


namespace equation_solution_l2503_250359

theorem equation_solution : ∃! x : ℚ, (3 * x - 15) / 4 = (x + 9) / 5 ∧ x = 10 := by sorry

end equation_solution_l2503_250359


namespace right_isosceles_triangle_special_property_l2503_250368

/-- A right isosceles triangle with the given property has 45° acute angles -/
theorem right_isosceles_triangle_special_property (a h : ℝ) (θ : ℝ) : 
  a > 0 → -- The leg length is positive
  h > 0 → -- The hypotenuse length is positive
  h = a * Real.sqrt 2 → -- Right isosceles triangle property
  h^2 = 3 * a * Real.sin θ → -- Given special property
  θ = π/4 := by -- Conclusion: acute angle is 45° (π/4 radians)
sorry

end right_isosceles_triangle_special_property_l2503_250368


namespace intersection_of_A_and_B_l2503_250303

def set_A : Set Int := {x | |x| < 3}
def set_B : Set Int := {x | |x| > 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {-2, 2} := by sorry

end intersection_of_A_and_B_l2503_250303


namespace solution_equals_expected_l2503_250399

-- Define the clubsuit operation
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the set of points satisfying x ⋆ y = y ⋆ x
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | clubsuit p.1 p.2 = clubsuit p.2 p.1}

-- Define the union of x-axis, y-axis, and lines y = x and y = -x
def expected_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2}

-- Theorem statement
theorem solution_equals_expected : solution_set = expected_set := by
  sorry

end solution_equals_expected_l2503_250399


namespace probability_factor_less_than_10_of_90_l2503_250302

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem probability_factor_less_than_10_of_90 :
  let all_factors := factors 90
  let factors_less_than_10 := all_factors.filter (λ x => x < 10)
  (factors_less_than_10.card : ℚ) / all_factors.card = 1 / 2 := by
  sorry

end probability_factor_less_than_10_of_90_l2503_250302


namespace keatons_annual_profit_l2503_250379

/-- Represents a fruit type with its harvest frequency, selling price, and cost --/
structure Fruit where
  harvestFrequency : Nat
  sellingPrice : Nat
  harvestCost : Nat

/-- Calculates the annual profit for a single fruit type --/
def annualProfit (fruit : Fruit) : Nat :=
  let harvestsPerYear := 12 / fruit.harvestFrequency
  let profitPerHarvest := fruit.sellingPrice - fruit.harvestCost
  harvestsPerYear * profitPerHarvest

/-- Keaton's farm setup --/
def oranges : Fruit := ⟨2, 50, 20⟩
def apples : Fruit := ⟨3, 30, 15⟩
def peaches : Fruit := ⟨4, 45, 25⟩
def blackberries : Fruit := ⟨6, 70, 30⟩

/-- Theorem: Keaton's total annual profit is $380 --/
theorem keatons_annual_profit :
  annualProfit oranges + annualProfit apples + annualProfit peaches + annualProfit blackberries = 380 := by
  sorry

end keatons_annual_profit_l2503_250379


namespace purely_imaginary_complex_number_l2503_250332

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - 1)
  (z.re = 0 ∧ z.im ≠ 0) → m = 3 := by
  sorry

end purely_imaginary_complex_number_l2503_250332


namespace total_football_games_l2503_250347

def football_games_this_year : ℕ := 4
def football_games_last_year : ℕ := 9

theorem total_football_games : 
  football_games_this_year + football_games_last_year = 13 := by
  sorry

end total_football_games_l2503_250347


namespace salary_increase_l2503_250386

theorem salary_increase (new_salary : ℝ) (increase_percentage : ℝ) (old_salary : ℝ) : 
  new_salary = 120 ∧ increase_percentage = 100 → old_salary = 60 := by
  sorry

end salary_increase_l2503_250386


namespace target_line_is_perpendicular_l2503_250393

/-- A line passing through a point and perpendicular to another line -/
def perpendicular_line (x y : ℝ) : Prop :=
  ∃ (A B C : ℝ), 
    (A * 3 + B * 4 + C = 0) ∧ 
    (A * B = -1) ∧
    (A * 2 + B * (-1) = 0)

/-- The specific line we're looking for -/
def target_line (x y : ℝ) : Prop :=
  x + 2 * y - 11 = 0

/-- Theorem stating that the target line satisfies the conditions -/
theorem target_line_is_perpendicular : 
  perpendicular_line 3 4 ↔ target_line 3 4 :=
sorry

end target_line_is_perpendicular_l2503_250393


namespace repeating_decimal_length_seven_twelfths_l2503_250365

theorem repeating_decimal_length_seven_twelfths :
  ∃ (d : ℕ) (n : ℕ), 
    7 * (10^n) ≡ d [MOD 12] ∧ 
    7 * (10^(n+1)) ≡ d [MOD 12] ∧ 
    0 < d ∧ d < 12 ∧
    n = 1 :=
by sorry

end repeating_decimal_length_seven_twelfths_l2503_250365


namespace tetrahedron_edge_relation_l2503_250358

/-- Given a tetrahedron ABCD with edge lengths and angles, prove that among t₁, t₂, t₃,
    there is at least one number equal to the sum of the other two. -/
theorem tetrahedron_edge_relation (a₁ a₂ a₃ b₁ b₂ b₃ θ₁ θ₂ θ₃ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0)
  (h_angle : 0 < θ₁ ∧ θ₁ < π ∧ 0 < θ₂ ∧ θ₂ < π ∧ 0 < θ₃ ∧ θ₃ < π) :
  ∃ (i j k : Fin 3), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    let t : Fin 3 → ℝ := λ n => match n with
      | 0 => a₁ * b₁ * Real.cos θ₁
      | 1 => a₂ * b₂ * Real.cos θ₂
      | 2 => a₃ * b₃ * Real.cos θ₃
    t i = t j + t k :=
by sorry

end tetrahedron_edge_relation_l2503_250358


namespace at_least_one_not_greater_than_third_l2503_250343

theorem at_least_one_not_greater_than_third (a b c : ℝ) (h : a + b + c = 1) :
  min a (min b c) ≤ 1/3 := by sorry

end at_least_one_not_greater_than_third_l2503_250343


namespace functional_equation_solution_l2503_250362

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → a * b * c * d = 1 →
    (f a + f b) * (f c + f d) = (a + b) * (c + d)

/-- The main theorem stating that any function satisfying the equation
    must be either the identity function or its reciprocal -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (hf : ∀ x : ℝ, x > 0 → f x > 0) 
    (heq : SatisfiesEquation f) :
    (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) := by
  sorry

end functional_equation_solution_l2503_250362


namespace special_hexagon_side_length_l2503_250382

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side_length : ℝ
  -- Three nonadjacent acute interior angles measure 45°
  has_45_degree_angles : Prop
  -- The enclosed area of the hexagon
  area : ℝ
  -- The hexagon is equilateral
  is_equilateral : Prop
  -- The area is 9√2
  area_is_9_sqrt_2 : area = 9 * Real.sqrt 2

/-- Theorem stating that a hexagon with the given properties has a side length of 2√3 -/
theorem special_hexagon_side_length 
  (h : SpecialHexagon) : h.side_length = 2 * Real.sqrt 3 := by
  sorry

end special_hexagon_side_length_l2503_250382


namespace reciprocal_of_sum_l2503_250328

theorem reciprocal_of_sum : (1 / (1/3 + 3/4)) = 12/13 := by sorry

end reciprocal_of_sum_l2503_250328


namespace sum_of_odd_numbers_sum_of_cubes_l2503_250354

def u (n : ℕ) : ℕ := (2 * n - 1) + if n = 0 then 0 else u (n - 1)

def S (n : ℕ) : ℕ := n^3 + if n = 0 then 0 else S (n - 1)

theorem sum_of_odd_numbers (n : ℕ) : u n = n^2 := by
  sorry

theorem sum_of_cubes (n : ℕ) : S n = (n * (n + 1) / 2)^2 := by
  sorry

end sum_of_odd_numbers_sum_of_cubes_l2503_250354


namespace driver_speed_proof_l2503_250376

theorem driver_speed_proof (v : ℝ) : v > 0 → v / (v + 12) = 2/3 → v = 24 := by
  sorry

end driver_speed_proof_l2503_250376


namespace max_faces_limited_neighbor_tri_neighbor_is_tetrahedron_l2503_250342

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler_formula : vertices - edges + faces = 2
  edge_face_relation : edges = 2 * faces

/-- A convex polyhedron where each face has at most 4 neighboring faces. -/
structure LimitedNeighborPolyhedron extends ConvexPolyhedron where
  max_neighbors : edges ≤ 2 * faces

/-- A convex polyhedron where each face has exactly 3 neighboring faces. -/
structure TriNeighborPolyhedron extends ConvexPolyhedron where
  tri_neighbors : edges = 3 * faces / 2

/-- Theorem: The maximum number of faces in a LimitedNeighborPolyhedron is 6. -/
theorem max_faces_limited_neighbor (P : LimitedNeighborPolyhedron) : P.faces ≤ 6 := by
  sorry

/-- Theorem: A TriNeighborPolyhedron must be a tetrahedron (4 faces). -/
theorem tri_neighbor_is_tetrahedron (P : TriNeighborPolyhedron) : P.faces = 4 := by
  sorry

end max_faces_limited_neighbor_tri_neighbor_is_tetrahedron_l2503_250342


namespace nested_fraction_equality_l2503_250348

theorem nested_fraction_equality : 
  1 + (1 / (1 + (1 / (2 + (1 / 3))))) = 17 / 10 := by
  sorry

end nested_fraction_equality_l2503_250348


namespace arithmetic_sequence_problem_l2503_250380

theorem arithmetic_sequence_problem (a d : ℤ) :
  let seq := [a, a + d, a + 2*d, a + 3*d, a + 4*d]
  (a^3 + (a + d)^3 + (a + 2*d)^3 + (a + 3*d)^3 = 16 * (a + (a + d) + (a + 2*d) + (a + 3*d))^2) ∧
  ((a + d)^3 + (a + 2*d)^3 + (a + 3*d)^3 + (a + 4*d)^3 = 16 * ((a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d))^2) →
  seq = [0, 16, 32, 48, 64] :=
by sorry

end arithmetic_sequence_problem_l2503_250380


namespace error_percentage_division_vs_multiplication_error_percentage_division_vs_multiplication_proof_l2503_250360

theorem error_percentage_division_vs_multiplication : ℝ → Prop :=
  fun x =>
    let correct_result := 2 * x
    let incorrect_result := x / 10
    let error := correct_result - incorrect_result
    let percentage_error := (error / correct_result) * 100
    percentage_error = 95

-- The proof is omitted
theorem error_percentage_division_vs_multiplication_proof :
  ∀ x : ℝ, x ≠ 0 → error_percentage_division_vs_multiplication x :=
sorry

end error_percentage_division_vs_multiplication_error_percentage_division_vs_multiplication_proof_l2503_250360


namespace equalize_payment_is_two_l2503_250387

/-- The amount B should give A to equalize payments when buying basketballs -/
def equalize_payment (n : ℕ+) : ℚ :=
  let total_cost := n.val * n.val
  let full_payments := total_cost / 10
  let remainder := total_cost % 10
  if remainder = 0 then 0
  else (10 - remainder) / 2

theorem equalize_payment_is_two (n : ℕ+) : equalize_payment n = 2 := by
  sorry


end equalize_payment_is_two_l2503_250387


namespace parabola_point_focus_distance_l2503_250300

theorem parabola_point_focus_distance (p m : ℝ) : 
  p > 0 → 
  m^2 = 2*p*4 → 
  (4 + p/2)^2 + m^2 = (17/4)^2 → 
  p = 1/2 ∧ (m = 2 ∨ m = -2) := by
sorry

end parabola_point_focus_distance_l2503_250300


namespace computer_price_increase_l2503_250353

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 580) : 
  d * (1 + 0.3) = 377 := by sorry

end computer_price_increase_l2503_250353


namespace josephs_birth_year_l2503_250390

-- Define the year of the first revised AMC 8
def first_revised_amc8_year : ℕ := 1987

-- Define Joseph's age when he took the seventh AMC 8
def josephs_age_at_seventh_amc8 : ℕ := 15

-- Define the number of years between the first and seventh AMC 8
def years_between_first_and_seventh_amc8 : ℕ := 6

-- Theorem to prove Joseph's birth year
theorem josephs_birth_year : 
  first_revised_amc8_year + years_between_first_and_seventh_amc8 - josephs_age_at_seventh_amc8 = 1978 := by
  sorry

end josephs_birth_year_l2503_250390


namespace stratified_sample_B_size_l2503_250395

/-- Represents the number of individuals in each level -/
structure PopulationLevels where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Calculates the total population -/
def total_population (p : PopulationLevels) : ℕ := p.A + p.B + p.C

/-- Represents a stratified sample -/
structure StratifiedSample where
  total_sample : ℕ
  population : PopulationLevels

/-- Calculates the number of individuals to be sampled from a specific level -/
def sample_size_for_level (s : StratifiedSample) (level_size : ℕ) : ℕ :=
  (s.total_sample * level_size) / (total_population s.population)

theorem stratified_sample_B_size 
  (sample : StratifiedSample) 
  (h1 : sample.population.A = 5 * n)
  (h2 : sample.population.B = 3 * n)
  (h3 : sample.population.C = 2 * n)
  (h4 : sample.total_sample = 150)
  (n : ℕ) :
  sample_size_for_level sample sample.population.B = 45 := by
  sorry

end stratified_sample_B_size_l2503_250395


namespace derivative_zero_at_origin_l2503_250330

theorem derivative_zero_at_origin (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f (-x) = f x) : 
  deriv f 0 = 0 := by
sorry

end derivative_zero_at_origin_l2503_250330


namespace edward_spent_sixteen_l2503_250351

def edward_book_purchase (initial_amount : ℕ) (remaining_amount : ℕ) (num_books : ℕ) : Prop :=
  ∃ (amount_spent : ℕ), 
    initial_amount = remaining_amount + amount_spent ∧
    amount_spent = 16

theorem edward_spent_sixteen : 
  edward_book_purchase 22 6 92 :=
sorry

end edward_spent_sixteen_l2503_250351


namespace sin_double_angle_special_point_l2503_250310

/-- Given an angle θ in standard position with its terminal side passing through the point (1, -2),
    prove that sin(2θ) = -4/5 -/
theorem sin_double_angle_special_point :
  ∀ θ : Real,
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = 1 ∧ r * Real.sin θ = -2) →
  Real.sin (2 * θ) = -4/5 := by
sorry

end sin_double_angle_special_point_l2503_250310


namespace sector_area_l2503_250363

/-- Given a circular sector with perimeter 10 and central angle 3 radians, its area is 6 -/
theorem sector_area (r : ℝ) (perimeter : ℝ) (central_angle : ℝ) : 
  perimeter = 10 → central_angle = 3 → perimeter = 2 * r + central_angle * r → 
  (1/2) * r^2 * central_angle = 6 := by
  sorry

end sector_area_l2503_250363


namespace sum_of_odd_periodic_function_l2503_250325

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x

theorem sum_of_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_periodic : has_period_4 f) 
  (h_f1 : f 1 = -1) : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
sorry

end sum_of_odd_periodic_function_l2503_250325


namespace tangent_line_slope_intersection_line_equation_l2503_250324

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define point Q
def Q : ℝ × ℝ := (0, -2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem for part 1
theorem tangent_line_slope :
  ∃ m : ℝ, m = -3/4 ∧
  (∀ x y : ℝ, y - P.2 = m * (x - P.1) → 
   (∃ t : ℝ, x = t ∧ y = t ∧ circle_C x y)) ∧
  (∀ x y : ℝ, circle_C x y → (y - P.2 ≠ m * (x - P.1) ∨ (x = P.1 ∧ y = P.2))) :=
sorry

-- Theorem for part 2
theorem intersection_line_equation :
  ∃ k : ℝ, (k = 5/3 ∨ k = 1) ∧
  (∀ x y : ℝ, y = k*x - 2 →
   (∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    y = k*x - 2 ∧
    (A.2 / A.1) * (B.2 / B.1) = -1/7)) :=
sorry

end tangent_line_slope_intersection_line_equation_l2503_250324


namespace ratio_x_to_y_l2503_250383

theorem ratio_x_to_y (x y : ℝ) (h : (15 * x - 4 * y) / (18 * x - 3 * y) = 4 / 7) :
  x / y = 16 / 33 := by
sorry

end ratio_x_to_y_l2503_250383


namespace collective_purchase_equation_l2503_250304

theorem collective_purchase_equation (x y : ℤ) : 
  (8 * x - 3 = y) → (7 * x + 4 = y) := by
  sorry

end collective_purchase_equation_l2503_250304


namespace shoes_count_l2503_250329

/-- The total number of pairs of shoes Ellie and Riley have together -/
def total_shoes (ellie_shoes : ℕ) (riley_difference : ℕ) : ℕ :=
  ellie_shoes + (ellie_shoes - riley_difference)

/-- Theorem stating that given Ellie has 8 pairs of shoes and Riley has 3 fewer pairs,
    they have 13 pairs of shoes in total -/
theorem shoes_count : total_shoes 8 3 = 13 := by
  sorry

end shoes_count_l2503_250329


namespace mothers_daughters_ages_l2503_250321

theorem mothers_daughters_ages (mother_age daughter_age : ℕ) : 
  mother_age = 40 →
  daughter_age + 2 * mother_age = 95 →
  mother_age + 2 * daughter_age = 70 := by
  sorry

end mothers_daughters_ages_l2503_250321


namespace chicken_selling_price_l2503_250338

/-- Represents the problem of determining the selling price of chickens --/
theorem chicken_selling_price 
  (num_chickens : ℕ) 
  (profit : ℚ) 
  (feed_per_chicken : ℚ) 
  (feed_bag_weight : ℚ) 
  (feed_bag_cost : ℚ) :
  num_chickens = 50 →
  profit = 65 →
  feed_per_chicken = 2 →
  feed_bag_weight = 20 →
  feed_bag_cost = 2 →
  ∃ (selling_price : ℚ), selling_price = 3/2 :=
by sorry

end chicken_selling_price_l2503_250338


namespace new_premium_calculation_l2503_250306

def calculate_new_premium (initial_premium : ℝ) (accident_increase_percent : ℝ) 
  (ticket_increase : ℝ) (late_payment_increase : ℝ) (num_accidents : ℕ) 
  (num_tickets : ℕ) (num_late_payments : ℕ) : ℝ :=
  initial_premium + 
  (initial_premium * accident_increase_percent * num_accidents : ℝ) +
  (ticket_increase * num_tickets) +
  (late_payment_increase * num_late_payments)

theorem new_premium_calculation :
  calculate_new_premium 125 0.12 7 15 2 4 3 = 228 := by
  sorry

end new_premium_calculation_l2503_250306
