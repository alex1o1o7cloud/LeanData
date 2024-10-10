import Mathlib

namespace exponent_division_l2947_294722

theorem exponent_division (a : ℝ) : a^5 / a^2 = a^3 := by
  sorry

end exponent_division_l2947_294722


namespace gwens_spent_money_l2947_294731

/-- Gwen's birthday money problem -/
theorem gwens_spent_money (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 5)
  (h2 : remaining_amount = 2) :
  initial_amount - remaining_amount = 3 := by
sorry

end gwens_spent_money_l2947_294731


namespace bryans_offer_l2947_294701

/-- Represents the problem of determining Bryan's offer for half of Peggy's record collection. -/
theorem bryans_offer (total_records : ℕ) (sammys_price : ℚ) (bryans_uninterested_price : ℚ) 
  (profit_difference : ℚ) (h1 : total_records = 200) (h2 : sammys_price = 4) 
  (h3 : bryans_uninterested_price = 1) (h4 : profit_difference = 100) : 
  ∃ (bryans_interested_price : ℚ),
    sammys_price * total_records - 
    (bryans_interested_price * (total_records / 2) + 
     bryans_uninterested_price * (total_records / 2)) = profit_difference ∧
    bryans_interested_price = 6 := by
  sorry

end bryans_offer_l2947_294701


namespace fourth_root_fifth_root_approx_l2947_294786

theorem fourth_root_fifth_root_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |((32 : ℝ) / 100000)^((1/5 : ℝ) * (1/4 : ℝ)) - 0.6687| < ε := by
  sorry

end fourth_root_fifth_root_approx_l2947_294786


namespace square_division_theorem_l2947_294793

theorem square_division_theorem (s : ℝ) :
  s > 0 →
  (3 * s = 42) →
  s = 14 :=
by sorry

end square_division_theorem_l2947_294793


namespace shortest_distance_line_to_circle_l2947_294760

/-- The shortest distance from a point on the line y = x + 1 to a point on the circle x^2 + y^2 + 2x + 4y + 4 = 0 is √2 - 1 -/
theorem shortest_distance_line_to_circle :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 4*p.2 + 4 = 0}
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ circle →
      Real.sqrt 2 - 1 ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) :=
by sorry

end shortest_distance_line_to_circle_l2947_294760


namespace hotel_rooms_l2947_294759

/-- Proves that a hotel with the given conditions has 10 rooms -/
theorem hotel_rooms (R : ℕ) 
  (people_per_room : ℕ) 
  (towels_per_person : ℕ) 
  (total_towels : ℕ) 
  (h1 : people_per_room = 3) 
  (h2 : towels_per_person = 2) 
  (h3 : total_towels = 60) 
  (h4 : R * people_per_room * towels_per_person = total_towels) : 
  R = 10 := by
  sorry

#check hotel_rooms

end hotel_rooms_l2947_294759


namespace simplify_expression_l2947_294720

theorem simplify_expression (b : ℝ) : (1)*(2*b)*(3*b^2)*(4*b^3)*(5*b^4)*(6*b^5) = 720*b^15 := by
  sorry

end simplify_expression_l2947_294720


namespace angle_between_lines_l2947_294772

def line1_direction : ℝ × ℝ := (2, 1)
def line2_direction : ℝ × ℝ := (4, 2)

theorem angle_between_lines (θ : ℝ) : 
  θ = Real.arccos (
    (line1_direction.1 * line2_direction.1 + line1_direction.2 * line2_direction.2) /
    (Real.sqrt (line1_direction.1^2 + line1_direction.2^2) * 
     Real.sqrt (line2_direction.1^2 + line2_direction.2^2))
  ) →
  Real.cos θ = 1 := by
sorry

end angle_between_lines_l2947_294772


namespace amusement_park_optimization_l2947_294712

/-- Represents the ticket cost and ride time for an attraction -/
structure Attraction where
  ticketCost : Nat
  rideTime : Float

/-- Represents a ticket purchase option -/
structure TicketOption where
  quantity : Nat
  price : Float

theorem amusement_park_optimization (budget : Float) 
  (ferrisWheel rollerCoaster bumperCars carousel hauntedHouse : Attraction)
  (entranceFee : Float) (initialTickets : Nat)
  (individualTicketPrice : Float) (tenTicketBundle twentyTicketBundle : TicketOption)
  (lunchMinCost lunchMaxCost : Float) (souvenirMinCost souvenirMaxCost : Float)
  (timeBeforeActivity activityDuration : Float) :
  budget = 50 ∧ 
  entranceFee = 10 ∧ initialTickets = 5 ∧
  ferrisWheel = { ticketCost := 5, rideTime := 0.3 } ∧
  rollerCoaster = { ticketCost := 4, rideTime := 0.3 } ∧
  bumperCars = { ticketCost := 4, rideTime := 0.3 } ∧
  carousel = { ticketCost := 3, rideTime := 0.3 } ∧
  hauntedHouse = { ticketCost := 6, rideTime := 0.3 } ∧
  individualTicketPrice = 1.5 ∧
  tenTicketBundle = { quantity := 10, price := 12 } ∧
  twentyTicketBundle = { quantity := 20, price := 22 } ∧
  lunchMinCost = 8 ∧ lunchMaxCost = 15 ∧
  souvenirMinCost = 5 ∧ souvenirMaxCost = 12 ∧
  timeBeforeActivity = 3 ∧ activityDuration = 1 →
  (∃ (optimalPurchase : TicketOption),
    optimalPurchase = twentyTicketBundle ∧
    ferrisWheel.rideTime + rollerCoaster.rideTime + bumperCars.rideTime + 
    carousel.rideTime + hauntedHouse.rideTime = 1.5) := by
  sorry

end amusement_park_optimization_l2947_294712


namespace existence_of_positive_reals_l2947_294777

theorem existence_of_positive_reals : ∃ (x y z : ℝ), 
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^4 + y^4 + z^4 = 13 ∧
  x^3*y^3*z + y^3*z^3*x + z^3*x^3*y = 6*Real.sqrt 3 ∧
  x^3*y*z + y^3*z*x + z^3*x*y = 5*Real.sqrt 3 :=
by sorry

end existence_of_positive_reals_l2947_294777


namespace train_speed_l2947_294752

/-- Proves that a train of given length crossing a bridge of given length in a given time travels at a specific speed. -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_l2947_294752


namespace geometric_sequence_ratio_l2947_294789

/-- Given a geometric sequence {a_n} where 2a₁, (3/2)a₂, a₃ form an arithmetic sequence,
    prove that the common ratio of the geometric sequence is either 1 or 2. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- {a_n} is a geometric sequence
  (2 * a 1 - (3/2 * a 2) = (3/2 * a 2) - a 3) →  -- 2a₁, (3/2)a₂, a₃ form an arithmetic sequence
  (a 2 / a 1 = 1 ∨ a 2 / a 1 = 2) :=
by sorry

end geometric_sequence_ratio_l2947_294789


namespace cards_per_page_l2947_294755

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 3) 
  (h2 : old_cards = 13) 
  (h3 : pages = 2) : 
  (new_cards + old_cards) / pages = 8 :=
by sorry

end cards_per_page_l2947_294755


namespace girls_on_track_l2947_294768

/-- Calculates the total number of girls on a track with given specifications -/
def total_girls (track_length : ℕ) (student_spacing : ℕ) : ℕ :=
  let students_per_side := track_length / student_spacing + 1
  let cycles_per_side := students_per_side / 3
  let girls_per_side := cycles_per_side * 2
  girls_per_side * 2

/-- The total number of girls on a 100-meter track with students every 2 meters,
    arranged in a pattern of two girls followed by one boy, is 68 -/
theorem girls_on_track : total_girls 100 2 = 68 := by
  sorry

end girls_on_track_l2947_294768


namespace perfect_square_characterization_l2947_294751

theorem perfect_square_characterization (A : ℕ+) :
  (∃ (d : ℕ+), A = d ^ 2) ↔
  (∀ (n : ℕ+), ∃ (j : ℕ+), j ≤ n ∧ (n ∣ ((A + j) ^ 2 - A))) :=
by sorry

end perfect_square_characterization_l2947_294751


namespace courtyard_length_l2947_294742

/-- The length of a rectangular courtyard given its width and paving stone information. -/
theorem courtyard_length (width : ℚ) (num_stones : ℕ) (stone_length stone_width : ℚ) :
  width = 33 / 2 →
  num_stones = 132 →
  stone_length = 5 / 2 →
  stone_width = 2 →
  (num_stones * stone_length * stone_width) / width = 40 :=
by sorry

end courtyard_length_l2947_294742


namespace sqrt_D_irrational_l2947_294702

/-- Given a real number x, D is defined as a² + b² + c², where a = x, b = x + 2, and c = a + b -/
def D (x : ℝ) : ℝ :=
  let a := x
  let b := x + 2
  let c := a + b
  a^2 + b^2 + c^2

/-- Theorem stating that the square root of D is always irrational for any real input x -/
theorem sqrt_D_irrational (x : ℝ) : Irrational (Real.sqrt (D x)) := by
  sorry

end sqrt_D_irrational_l2947_294702


namespace apple_cost_price_l2947_294770

/-- 
Given:
- The selling price of an apple is 18
- The seller loses 1/6th of the cost price
Prove that the cost price is 21.6
-/
theorem apple_cost_price (selling_price : ℝ) (loss_fraction : ℝ) (cost_price : ℝ) : 
  selling_price = 18 → 
  loss_fraction = 1/6 → 
  selling_price = cost_price * (1 - loss_fraction) →
  cost_price = 21.6 := by
sorry

end apple_cost_price_l2947_294770


namespace monster_perimeter_l2947_294763

theorem monster_perimeter (r : ℝ) (θ : ℝ) (h1 : r = 2) (h2 : θ = 2 * π / 3) :
  let arc_length := (2 * π - θ) / (2 * π) * (2 * π * r)
  let chord_length := 2 * r * Real.sin (θ / 2)
  arc_length + chord_length = 8 * π / 3 + 2 * Real.sqrt 3 := by
  sorry

end monster_perimeter_l2947_294763


namespace tangent_line_equation_l2947_294721

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2 * Real.log x - 1/2

def a : ℝ := 2

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - 1 = 0) :=
by sorry

end tangent_line_equation_l2947_294721


namespace f_neg_one_lt_f_one_l2947_294747

/-- A function f: ℝ → ℝ that satisfies the given conditions -/
def f : ℝ → ℝ := sorry

/-- f is differentiable on ℝ -/
axiom f_differentiable : Differentiable ℝ f

/-- The functional equation for f -/
axiom f_eq (x : ℝ) : f x = x^2 + 2 * x * (deriv f 2)

/-- Theorem: f(-1) < f(1) -/
theorem f_neg_one_lt_f_one : f (-1) < f 1 := by sorry

end f_neg_one_lt_f_one_l2947_294747


namespace log_inequality_sufficiency_not_necessity_l2947_294705

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Statement of the theorem
theorem log_inequality_sufficiency_not_necessity :
  (∀ a b : ℝ, log10 a > log10 b → a > b) ∧
  (∃ a b : ℝ, a > b ∧ ¬(log10 a > log10 b)) :=
by sorry

end log_inequality_sufficiency_not_necessity_l2947_294705


namespace max_xy_value_l2947_294794

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) :
  x * y ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    4 * x₀^2 + 9 * y₀^2 + 3 * x₀ * y₀ = 30 ∧ x₀ * y₀ = 2 :=
by sorry

end max_xy_value_l2947_294794


namespace positive_number_inequality_l2947_294783

theorem positive_number_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x y, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧
  0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) ≤ 1 :=
by sorry

end positive_number_inequality_l2947_294783


namespace tangent_line_derivative_l2947_294735

variable (f : ℝ → ℝ)

theorem tangent_line_derivative (h : ∀ y, y = (1/2) * 1 + 3 → y = f 1) :
  deriv f 1 = 1/2 := by sorry

end tangent_line_derivative_l2947_294735


namespace jillian_shells_l2947_294743

theorem jillian_shells (savannah_shells clayton_shells : ℕ) 
  (h1 : savannah_shells = 17)
  (h2 : clayton_shells = 8)
  (h3 : ∃ (total_shells : ℕ), total_shells = 27 * 2)
  (h4 : ∃ (jillian_shells : ℕ), jillian_shells + savannah_shells + clayton_shells = 27 * 2) :
  ∃ (jillian_shells : ℕ), jillian_shells = 29 := by
sorry

end jillian_shells_l2947_294743


namespace parallel_line_plane_false_l2947_294700

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem to be proven false
theorem parallel_line_plane_false :
  ¬(∀ (l : Line) (p : Plane), parallel_plane l p →
    ∀ (m : Line), contained_in m p → parallel_line l m) :=
sorry

end parallel_line_plane_false_l2947_294700


namespace profit_percentage_is_20_percent_l2947_294727

def selling_price : ℚ := 1170
def cost_price : ℚ := 975

theorem profit_percentage_is_20_percent :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end profit_percentage_is_20_percent_l2947_294727


namespace function_range_implies_a_value_l2947_294726

theorem function_range_implies_a_value (a : ℝ) (h1 : a > 0) : 
  (∀ x ∈ Set.Icc a (2 * a), (8 / x) ∈ Set.Icc (a / 4) 2) → a = 4 := by
  sorry

end function_range_implies_a_value_l2947_294726


namespace absolute_value_equation_solution_l2947_294716

theorem absolute_value_equation_solution :
  ∃! y : ℚ, |y - 3| = |y + 2| :=
by
  -- The proof goes here
  sorry

end absolute_value_equation_solution_l2947_294716


namespace circle_tangent_to_axes_l2947_294733

/-- A circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The standard equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A circle is tangent to the x-axis -/
def Circle.tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

/-- A circle is tangent to the y-axis -/
def Circle.tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

/-- The main theorem -/
theorem circle_tangent_to_axes (c : Circle) :
  c.radius = 2 ∧ c.tangentToXAxis ∧ c.tangentToYAxis ↔ 
  ∀ x y : ℝ, c.equation x y ↔ (x - 2)^2 + (y - 2)^2 = 4 :=
sorry

end circle_tangent_to_axes_l2947_294733


namespace arithmetic_geometric_sequence_problem_l2947_294798

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_problem 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)
  (h_condition : a 6 - (a 7)^2 + a 8 = 0)
  (h_geometric : geometric_sequence b)
  (h_equal : b 7 = a 7) :
  b 3 * b 8 * b 10 = 8 := by
sorry

end arithmetic_geometric_sequence_problem_l2947_294798


namespace chameleon_color_impossibility_l2947_294775

/-- Represents the state of chameleons on the island -/
structure ChameleonSystem :=
  (num_chameleons : Nat)
  (num_colors : Nat)
  (color_change : Nat → Nat → Nat)  -- Function representing color change

/-- Represents the property that all chameleons have been all colors -/
def all_chameleons_all_colors (system : ChameleonSystem) : Prop :=
  ∀ c : Nat, c < system.num_chameleons → 
    ∃ t1 t2 t3 : Nat, 
      system.color_change c t1 = 0 ∧ 
      system.color_change c t2 = 1 ∧ 
      system.color_change c t3 = 2

theorem chameleon_color_impossibility :
  ∀ system : ChameleonSystem, 
    system.num_chameleons = 35 → 
    system.num_colors = 3 → 
    ¬(all_chameleons_all_colors system) := by
  sorry

end chameleon_color_impossibility_l2947_294775


namespace polynomial_division_quotient_l2947_294728

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 12 * X^3 + 24 * X^2 - 10 * X + 5
  let divisor : Polynomial ℚ := 3 * X + 4
  let quotient : Polynomial ℚ := 4 * X^2 - 22/3
  dividend = divisor * quotient + (Polynomial.C (-197/9) : Polynomial ℚ) := by sorry

end polynomial_division_quotient_l2947_294728


namespace expression_equals_36_l2947_294761

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end expression_equals_36_l2947_294761


namespace min_value_trigonometric_function_l2947_294740

theorem min_value_trigonometric_function :
  ∀ x : ℝ, 0 < x → x < π / 2 →
    1 / (Real.sin x)^2 + 12 * Real.sqrt 3 / Real.cos x ≥ 28 ∧
    ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < π / 2 ∧
      1 / (Real.sin x₀)^2 + 12 * Real.sqrt 3 / Real.cos x₀ = 28 := by
  sorry

end min_value_trigonometric_function_l2947_294740


namespace calculate_expression_l2947_294758

theorem calculate_expression : 7 * (9 + 2/5) + 3 = 68.8 := by
  sorry

end calculate_expression_l2947_294758


namespace prime_sequence_l2947_294714

theorem prime_sequence (A : ℕ) : 
  Nat.Prime A ∧ 
  Nat.Prime (A - 4) ∧ 
  Nat.Prime (A - 6) ∧ 
  Nat.Prime (A - 12) ∧ 
  Nat.Prime (A - 18) → 
  A = 23 := by
sorry

end prime_sequence_l2947_294714


namespace equal_roots_quadratic_l2947_294795

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ (∀ y : ℝ, y^2 + 2*y + k = 0 → y = x)) → k = 1 := by
  sorry

end equal_roots_quadratic_l2947_294795


namespace essay_word_count_excess_l2947_294732

theorem essay_word_count_excess (word_limit : ℕ) (saturday_words : ℕ) (sunday_words : ℕ) :
  word_limit = 1000 →
  saturday_words = 450 →
  sunday_words = 650 →
  (saturday_words + sunday_words) - word_limit = 100 := by
  sorry

end essay_word_count_excess_l2947_294732


namespace floor_tiles_l2947_294749

theorem floor_tiles (black_tiles : ℕ) (total_tiles : ℕ) : 
  black_tiles = 441 → 
  ∃ (side_length : ℕ), 
    side_length * side_length = total_tiles ∧
    side_length = (black_tiles.sqrt : ℕ) + 2 * 3 →
    total_tiles = 729 :=
by sorry

end floor_tiles_l2947_294749


namespace coeff_x_squared_in_expansion_l2947_294711

/-- The coefficient of x^2 in the expansion of (1+2x)^6 is 60 -/
theorem coeff_x_squared_in_expansion : 
  (Finset.range 7).sum (fun k => (Nat.choose 6 k) * (1^(6-k)) * ((2:ℕ)^k)) = 60 := by
  sorry

end coeff_x_squared_in_expansion_l2947_294711


namespace polynomial_simplification_l2947_294757

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 - 3 * q^3 + 7 * q - 8) + (5 - 2 * q^3 + 9 * q^2 - 4 * q) =
  4 * q^4 - 5 * q^3 + 9 * q^2 + 3 * q - 3 := by
  sorry

end polynomial_simplification_l2947_294757


namespace condition_neither_necessary_nor_sufficient_l2947_294704

-- Define the sets M and P
def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

-- Statement to prove
theorem condition_neither_necessary_nor_sufficient :
  ¬(∀ x : ℝ, (x ∈ M ∩ P → (x ∈ M ∨ x ∈ P)) ∧ ((x ∈ M ∨ x ∈ P) → x ∈ M ∩ P)) :=
sorry

end condition_neither_necessary_nor_sufficient_l2947_294704


namespace range_of_a_range_of_m_l2947_294741

-- Part 1
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*a*(x-1) + 4) →
  a ∈ Set.Iic (1/3) :=
sorry

-- Part 2
def f' (x : ℝ) : ℝ := x^2 - 3*x + 2
def g (m : ℝ) (x : ℝ) : ℝ := -x + m

theorem range_of_m :
  ∀ m : ℝ, (∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Ioo 1 8, f' x₁ = g m x₂) →
  m ∈ Set.Ioo 7 (31/4) :=
sorry

end range_of_a_range_of_m_l2947_294741


namespace octagon_area_eq_1200_l2947_294706

/-- A regular octagon inscribed in a square with perimeter 160 cm,
    where each side of the square is quadrised by the vertices of the octagon -/
structure InscribedOctagon where
  square_perimeter : ℝ
  square_perimeter_eq : square_perimeter = 160
  is_regular : Bool
  is_inscribed : Bool
  sides_quadrised : Bool

/-- The area of the inscribed octagon -/
def octagon_area (o : InscribedOctagon) : ℝ := sorry

/-- Theorem stating that the area of the inscribed octagon is 1200 square centimeters -/
theorem octagon_area_eq_1200 (o : InscribedOctagon) :
  o.is_regular ∧ o.is_inscribed ∧ o.sides_quadrised → octagon_area o = 1200 := by sorry

end octagon_area_eq_1200_l2947_294706


namespace cubic_roots_sum_min_l2947_294776

theorem cubic_roots_sum_min (a : ℝ) (x₁ x₂ x₃ : ℝ) (h_pos : a > 0) 
  (h_roots : x₁^3 - a*x₁^2 + a*x₁ - a = 0 ∧ 
             x₂^3 - a*x₂^2 + a*x₂ - a = 0 ∧ 
             x₃^3 - a*x₃^2 + a*x₃ - a = 0) : 
  ∃ (m : ℝ), m = -4 ∧ ∀ (y : ℝ), y ≥ m → x₁^3 + x₂^3 + x₃^3 - 3*x₁*x₂*x₃ ≥ y :=
sorry

end cubic_roots_sum_min_l2947_294776


namespace inequality_solution_l2947_294773

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 9 ↔ x < -3 ∨ x > 1 := by sorry

end inequality_solution_l2947_294773


namespace remaining_coin_value_l2947_294791

def initial_quarters : Nat := 11
def initial_dimes : Nat := 15
def initial_nickels : Nat := 7

def purchased_quarters : Nat := 1
def purchased_dimes : Nat := 8
def purchased_nickels : Nat := 3

def quarter_value : Nat := 25
def dime_value : Nat := 10
def nickel_value : Nat := 5

theorem remaining_coin_value :
  (initial_quarters - purchased_quarters) * quarter_value +
  (initial_dimes - purchased_dimes) * dime_value +
  (initial_nickels - purchased_nickels) * nickel_value = 340 := by
  sorry

end remaining_coin_value_l2947_294791


namespace train_length_l2947_294713

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5/18) →
  platform_length = 210 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 310 := by
sorry

end train_length_l2947_294713


namespace complement_M_in_U_l2947_294784

-- Define the set U
def U : Set ℕ := {1,2,3,4,5,6,7}

-- Define the set M
def M : Set ℕ := {x ∈ U | x^2 - 6*x + 5 ≤ 0}

-- State the theorem
theorem complement_M_in_U : (U \ M) = {6,7} := by sorry

end complement_M_in_U_l2947_294784


namespace solve_for_d_l2947_294792

theorem solve_for_d (c a m d : ℝ) (h : m = (c * a * d) / (a - d)) : 
  d = (m * a) / (m + c * a) := by
sorry

end solve_for_d_l2947_294792


namespace mean_median_difference_l2947_294745

/-- Represents the distribution of scores in a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_60_percent : ℚ
  score_75_percent : ℚ
  score_82_percent : ℚ
  score_88_percent : ℚ
  score_92_percent : ℚ

/-- Calculates the mean score given a score distribution -/
def mean_score (d : ScoreDistribution) : ℚ :=
  (60 * d.score_60_percent + 75 * d.score_75_percent + 82 * d.score_82_percent +
   88 * d.score_88_percent + 92 * d.score_92_percent) / 1

/-- Calculates the median score given a score distribution -/
def median_score (d : ScoreDistribution) : ℚ := 82

/-- Theorem stating the difference between mean and median scores -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.total_students = 30)
  (h2 : d.score_60_percent = 15/100)
  (h3 : d.score_75_percent = 20/100)
  (h4 : d.score_82_percent = 25/100)
  (h5 : d.score_88_percent = 30/100)
  (h6 : d.score_92_percent = 10/100) :
  mean_score d - median_score d = 47/100 := by
  sorry

end mean_median_difference_l2947_294745


namespace largest_prime_factor_of_b_16_minus_1_l2947_294717

-- Define b as a natural number
def b : ℕ := 2

-- Define the function for the number of distinct prime factors
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

-- Define the function for the largest prime factor
def largest_prime_factor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_b_16_minus_1 :
  num_distinct_prime_factors (b^16 - 1) = 4 →
  largest_prime_factor (b^16 - 1) = 257 :=
by sorry

end largest_prime_factor_of_b_16_minus_1_l2947_294717


namespace difference_of_squares_l2947_294774

theorem difference_of_squares (x y : ℝ) : (-x + y) * (x + y) = y^2 - x^2 := by
  sorry

end difference_of_squares_l2947_294774


namespace burj_khalifa_height_l2947_294708

theorem burj_khalifa_height (sears_height burj_difference : ℕ) 
  (h1 : sears_height = 527)
  (h2 : burj_difference = 303) : 
  sears_height + burj_difference = 830 := by
sorry

end burj_khalifa_height_l2947_294708


namespace sons_age_is_eighteen_l2947_294754

/-- Proves that the son's age is 18 years given the conditions in the problem -/
theorem sons_age_is_eighteen (son_age man_age : ℕ) : 
  man_age = son_age + 20 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
  sorry

end sons_age_is_eighteen_l2947_294754


namespace modulus_of_z_l2947_294765

def i : ℂ := Complex.I

def z : ℂ := (1 + i) * (1 + 2*i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_of_z_l2947_294765


namespace angle_in_first_quadrant_l2947_294782

theorem angle_in_first_quadrant (α : Real) (h : Real.sin α + Real.cos α > 1) : 
  0 < α ∧ α < Real.pi / 2 := by
  sorry

end angle_in_first_quadrant_l2947_294782


namespace max_value_complex_expression_l2947_294779

theorem max_value_complex_expression (x y : ℂ) :
  ∃ (M : ℝ), M = (5 * Real.sqrt 2) / 2 ∧
  Complex.abs (3 * x + 4 * y) / Real.sqrt (Complex.abs x ^ 2 + Complex.abs y ^ 2 + Complex.abs (x ^ 2 + y ^ 2)) ≤ M ∧
  ∃ (x₀ y₀ : ℂ), Complex.abs (3 * x₀ + 4 * y₀) / Real.sqrt (Complex.abs x₀ ^ 2 + Complex.abs y₀ ^ 2 + Complex.abs (x₀ ^ 2 + y₀ ^ 2)) = M :=
by sorry

end max_value_complex_expression_l2947_294779


namespace prob_heart_or_king_two_draws_l2947_294753

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either hearts or kings -/
def heart_or_king : ℕ := 16

/-- The probability of drawing a card that is neither a heart nor a king -/
def prob_not_heart_or_king : ℚ := (deck_size - heart_or_king) / deck_size

/-- The probability of drawing at least one heart or king in two draws with replacement -/
def prob_at_least_one_heart_or_king : ℚ := 1 - prob_not_heart_or_king ^ 2

theorem prob_heart_or_king_two_draws :
  prob_at_least_one_heart_or_king = 88 / 169 := by
  sorry

end prob_heart_or_king_two_draws_l2947_294753


namespace daisy_count_l2947_294719

def white_daisies : ℕ := 6

def pink_daisies : ℕ := 9 * white_daisies

def red_daisies : ℕ := 4 * pink_daisies - 3

def total_daisies : ℕ := white_daisies + pink_daisies + red_daisies

theorem daisy_count : total_daisies = 273 := by
  sorry

end daisy_count_l2947_294719


namespace adjacent_angle_measure_l2947_294715

-- Define the angle type
def Angle : Type := ℝ

-- Define parallel lines
def ParallelLines (m n : Line) : Prop := sorry

-- Define a transversal line
def Transversal (p m n : Line) : Prop := sorry

-- Define the measure of an angle
def AngleMeasure (θ : Angle) : ℝ := sorry

-- Define supplementary angles
def Supplementary (θ₁ θ₂ : Angle) : Prop :=
  AngleMeasure θ₁ + AngleMeasure θ₂ = 180

-- Theorem statement
theorem adjacent_angle_measure
  (m n p : Line)
  (θ₁ θ₂ : Angle)
  (h_parallel : ParallelLines m n)
  (h_transversal : Transversal p m n)
  (h_internal : AngleMeasure θ₁ = 70)
  (h_supplementary : Supplementary θ₁ θ₂) :
  AngleMeasure θ₂ = 110 :=
sorry

end adjacent_angle_measure_l2947_294715


namespace roller_coaster_probability_l2947_294739

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 4

/-- The number of times the passenger rides the roller coaster -/
def num_rides : ℕ := 3

/-- The probability of choosing a different car on the second ride -/
def prob_second_ride : ℚ := 3 / 4

/-- The probability of choosing a different car on the third ride -/
def prob_third_ride : ℚ := 1 / 2

/-- The probability of riding in 3 different cars over 3 rides -/
def prob_three_different_cars : ℚ := 3 / 8

theorem roller_coaster_probability :
  prob_three_different_cars = 1 * prob_second_ride * prob_third_ride :=
sorry

end roller_coaster_probability_l2947_294739


namespace makeup_fraction_of_savings_l2947_294723

/-- Given Leila's original savings and the cost of a sweater, prove the fraction spent on make-up -/
theorem makeup_fraction_of_savings (original_savings : ℚ) (sweater_cost : ℚ) 
  (h1 : original_savings = 80)
  (h2 : sweater_cost = 20) :
  (original_savings - sweater_cost) / original_savings = 3/4 := by
  sorry

end makeup_fraction_of_savings_l2947_294723


namespace hypotenuse_length_l2947_294736

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- First leg
  b : ℝ  -- Second leg
  c : ℝ  -- Hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) 
  (perimeter : t.a + t.b + t.c = 40)  -- Perimeter condition
  (area : (1/2) * t.a * t.b = 24)     -- Area condition
  : t.c = 18.8 := by
  sorry  -- Proof omitted

end hypotenuse_length_l2947_294736


namespace smallest_cube_volume_for_ziggurat_model_l2947_294744

/-- The volume of the smallest cube that can contain a rectangular prism -/
theorem smallest_cube_volume_for_ziggurat_model (h : ℕ) (b : ℕ) : 
  h = 15 → b = 8 → (max h b) ^ 3 = 3375 := by sorry

end smallest_cube_volume_for_ziggurat_model_l2947_294744


namespace sqrt_72_equals_6_sqrt_2_l2947_294762

theorem sqrt_72_equals_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_72_equals_6_sqrt_2_l2947_294762


namespace white_balls_count_prob_after_addition_l2947_294767

/-- The total number of balls in the box -/
def total_balls : ℕ := 40

/-- The probability of picking a white ball -/
def prob_white : ℚ := 1/10 * 6

/-- The number of white balls in the box -/
def white_balls : ℕ := 24

/-- The number of additional balls added -/
def additional_balls : ℕ := 10

/-- Theorem stating the relationship between the number of white balls and the probability -/
theorem white_balls_count : white_balls = total_balls * prob_white := by sorry

/-- Theorem proving that adding 10 balls with 1 white results in 50% probability -/
theorem prob_after_addition : 
  (white_balls + 1) / (total_balls + additional_balls) = 1/2 := by sorry

end white_balls_count_prob_after_addition_l2947_294767


namespace smallest_a1_l2947_294738

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 13 * a (n - 1) - 2 * n)

/-- The theorem stating the smallest possible value of a₁ -/
theorem smallest_a1 (a : ℕ → ℝ) (h : RecurrenceSequence a) :
  ∀ a₁ : ℝ, a 1 = a₁ → a₁ ≥ 13 / 36 :=
sorry

end smallest_a1_l2947_294738


namespace smallest_angle_equation_l2947_294718

/-- The smallest positive angle θ in degrees that satisfies the equation
    cos θ = sin 45° + cos 60° - sin 30° - cos 15° -/
theorem smallest_angle_equation : ∃ θ : ℝ,
  θ > 0 ∧
  θ < 360 ∧
  Real.cos (θ * π / 180) = Real.sin (45 * π / 180) + Real.cos (60 * π / 180) - 
                           Real.sin (30 * π / 180) - Real.cos (15 * π / 180) ∧
  ∀ φ, 0 < φ ∧ φ < θ → 
    Real.cos (φ * π / 180) ≠ Real.sin (45 * π / 180) + Real.cos (60 * π / 180) - 
                             Real.sin (30 * π / 180) - Real.cos (15 * π / 180) ∧
  θ = 30 :=
by sorry

end smallest_angle_equation_l2947_294718


namespace zachary_initial_money_l2947_294764

/-- Calculates Zachary's initial money given the costs of items and additional amount needed --/
theorem zachary_initial_money 
  (football_cost shorts_cost shoes_cost additional_needed : ℚ) 
  (h1 : football_cost = 3.75)
  (h2 : shorts_cost = 2.40)
  (h3 : shoes_cost = 11.85)
  (h4 : additional_needed = 8) :
  football_cost + shorts_cost + shoes_cost - additional_needed = 9 := by
sorry

end zachary_initial_money_l2947_294764


namespace fruit_remaining_l2947_294730

-- Define the quantities of fruits picked and eaten
def mike_apples : ℝ := 7.0
def nancy_apples : ℝ := 3.0
def john_apples : ℝ := 5.0
def keith_apples : ℝ := 6.0
def lisa_apples : ℝ := 2.0
def oranges_picked_and_eaten : ℝ := 8.0
def cherries_picked_and_eaten : ℝ := 4.0

-- Define the total apples picked and eaten
def total_apples_picked : ℝ := mike_apples + nancy_apples + john_apples
def total_apples_eaten : ℝ := keith_apples + lisa_apples

-- Theorem statement
theorem fruit_remaining :
  (total_apples_picked - total_apples_eaten = 7.0) ∧
  (oranges_picked_and_eaten - oranges_picked_and_eaten = 0) ∧
  (cherries_picked_and_eaten - cherries_picked_and_eaten = 0) :=
by sorry

end fruit_remaining_l2947_294730


namespace intersection_empty_iff_a_in_range_B_subset_A_iff_a_in_range_l2947_294724

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| ≤ 4}
def B : Set ℝ := {x : ℝ | (x - 2) * (x - 3) ≤ 0}

-- Theorem 1: A ∩ B = ∅ if and only if a ∈ (-∞, -2) ∪ (7, +∞)
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a < -2 ∨ a > 7 := by sorry

-- Theorem 2: B ⊆ A if and only if a ∈ [1, 6]
theorem B_subset_A_iff_a_in_range (a : ℝ) :
  B ⊆ A a ↔ 1 ≤ a ∧ a ≤ 6 := by sorry

end intersection_empty_iff_a_in_range_B_subset_A_iff_a_in_range_l2947_294724


namespace least_six_digit_congruent_to_seven_mod_seventeen_l2947_294766

theorem least_six_digit_congruent_to_seven_mod_seventeen :
  ∃ (n : ℕ), 
    n = 100008 ∧ 
    n ≥ 100000 ∧ 
    n < 1000000 ∧
    n % 17 = 7 ∧
    ∀ (m : ℕ), m ≥ 100000 ∧ m < 1000000 ∧ m % 17 = 7 → m ≥ n :=
by sorry

end least_six_digit_congruent_to_seven_mod_seventeen_l2947_294766


namespace f_has_two_zeros_l2947_294788

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x * Real.sin x

theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁ ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧
  x₂ ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧
  f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ f x = 0 → (x = x₁ ∨ x = x₂) :=
sorry

end f_has_two_zeros_l2947_294788


namespace stationery_cost_l2947_294769

/-- The total cost of a pen, pencil, and eraser with given price relationships -/
theorem stationery_cost (pencil_cost : ℚ) : 
  pencil_cost = 8 →
  (pencil_cost + (1/2 * pencil_cost) + (2 * (1/2 * pencil_cost))) = 20 := by
  sorry

end stationery_cost_l2947_294769


namespace exists_alternating_coloring_l2947_294707

-- Define an ordered set
variable {X : Type*} [PartialOrder X]

-- Define a coloring function
def Coloring (X : Type*) := X → Bool

-- Theorem statement
theorem exists_alternating_coloring :
  ∃ (f : Coloring X), ∀ (x y : X), x < y → f x = f y →
    ∃ (z : X), x < z ∧ z < y ∧ f z ≠ f x := by
  sorry

end exists_alternating_coloring_l2947_294707


namespace train_tunnel_crossing_time_l2947_294746

theorem train_tunnel_crossing_time
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (tunnel_length : ℝ)
  (h1 : train_length = 100)
  (h2 : train_speed_kmph = 72)
  (h3 : tunnel_length = 1400) :
  (train_length + tunnel_length) / (train_speed_kmph * (1000 / 3600)) = 75 :=
by sorry

end train_tunnel_crossing_time_l2947_294746


namespace elisa_family_women_without_daughters_l2947_294797

/-- Represents a family tree starting from Elisa -/
structure ElisaFamily where
  daughters : Nat
  granddaughters : Nat
  daughters_with_children : Nat

/-- The conditions of Elisa's family -/
def elisa_family : ElisaFamily where
  daughters := 8
  granddaughters := 28
  daughters_with_children := 4

/-- The total number of daughters and granddaughters -/
def total_descendants (f : ElisaFamily) : Nat :=
  f.daughters + f.granddaughters

/-- The number of women (daughters and granddaughters) who have no daughters -/
def women_without_daughters (f : ElisaFamily) : Nat :=
  (f.daughters - f.daughters_with_children) + f.granddaughters

/-- Theorem stating that 32 of Elisa's daughters and granddaughters have no daughters -/
theorem elisa_family_women_without_daughters :
  women_without_daughters elisa_family = 32 := by
  sorry

end elisa_family_women_without_daughters_l2947_294797


namespace decimal_division_l2947_294748

theorem decimal_division : (0.1 : ℝ) / 0.004 = 25 := by
  sorry

end decimal_division_l2947_294748


namespace decimal_to_percentage_example_l2947_294710

/-- Converts a decimal fraction to a percentage -/
def decimal_to_percentage (d : ℝ) : ℝ := d * 100

/-- The decimal fraction we're working with -/
def given_decimal : ℝ := 0.01

/-- Theorem stating that converting 0.01 to a percentage results in 1 -/
theorem decimal_to_percentage_example :
  decimal_to_percentage given_decimal = 1 := by
  sorry

end decimal_to_percentage_example_l2947_294710


namespace circle_radius_is_one_l2947_294750

/-- The radius of the circle with equation 16x^2 + 32x + 16y^2 - 48y + 68 = 0 is 1 -/
theorem circle_radius_is_one :
  ∃ (h k r : ℝ), r = 1 ∧
  ∀ (x y : ℝ), 16*x^2 + 32*x + 16*y^2 - 48*y + 68 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end circle_radius_is_one_l2947_294750


namespace base_seven_digits_of_956_l2947_294771

theorem base_seven_digits_of_956 : ∃ n : ℕ, (7^(n-1) ≤ 956 ∧ 956 < 7^n) ∧ n = 4 := by
  sorry

end base_seven_digits_of_956_l2947_294771


namespace original_number_before_increase_l2947_294796

theorem original_number_before_increase (final_number : ℝ) (increase_percentage : ℝ) (original_number : ℝ) : 
  final_number = 90 ∧ 
  increase_percentage = 50 ∧ 
  final_number = original_number * (1 + increase_percentage / 100) → 
  original_number = 60 := by
sorry

end original_number_before_increase_l2947_294796


namespace lawnmower_value_drop_l2947_294778

/-- Calculates the final value of a lawnmower after three successive value drops -/
theorem lawnmower_value_drop (initial_value : ℝ) (drop1 drop2 drop3 : ℝ) :
  initial_value = 100 →
  drop1 = 0.25 →
  drop2 = 0.20 →
  drop3 = 0.15 →
  initial_value * (1 - drop1) * (1 - drop2) * (1 - drop3) = 51 := by
  sorry

end lawnmower_value_drop_l2947_294778


namespace factorization_equality_l2947_294756

theorem factorization_equality (m n : ℝ) : m^2 * n - m * n = m * n * (m - 1) := by
  sorry

end factorization_equality_l2947_294756


namespace rectangle_perimeter_l2947_294729

-- Define the triangle
def triangle_side_1 : ℝ := 9
def triangle_side_2 : ℝ := 12
def triangle_hypotenuse : ℝ := 15

-- Define the rectangle
def rectangle_length : ℝ := 6

-- Theorem statement
theorem rectangle_perimeter : 
  -- Right triangle condition
  triangle_side_1^2 + triangle_side_2^2 = triangle_hypotenuse^2 →
  -- Rectangle area equals triangle area
  (1/2 * triangle_side_1 * triangle_side_2) = (rectangle_length * (1/2 * triangle_side_1 * triangle_side_2 / rectangle_length)) →
  -- Perimeter of the rectangle is 30
  2 * (rectangle_length + (1/2 * triangle_side_1 * triangle_side_2 / rectangle_length)) = 30 :=
by
  sorry

end rectangle_perimeter_l2947_294729


namespace time_to_top_floor_l2947_294790

/-- The number of floors in the building -/
def num_floors : ℕ := 10

/-- The time in seconds to go up to an even-numbered floor -/
def even_floor_time : ℕ := 15

/-- The time in seconds to go up to an odd-numbered floor -/
def odd_floor_time : ℕ := 9

/-- The number of even-numbered floors -/
def num_even_floors : ℕ := num_floors / 2

/-- The number of odd-numbered floors -/
def num_odd_floors : ℕ := (num_floors + 1) / 2

/-- The total time in seconds to reach the top floor -/
def total_time_seconds : ℕ := num_even_floors * even_floor_time + num_odd_floors * odd_floor_time

/-- Conversion factor from seconds to minutes -/
def seconds_per_minute : ℕ := 60

theorem time_to_top_floor :
  total_time_seconds / seconds_per_minute = 2 := by
  sorry

end time_to_top_floor_l2947_294790


namespace diamonds_count_l2947_294787

/-- Represents the number of gems in a treasure chest. -/
def total_gems : ℕ := 5155

/-- Represents the number of rubies in the treasure chest. -/
def rubies : ℕ := 5110

/-- Theorem stating that the number of diamonds in the treasure chest is 45. -/
theorem diamonds_count : total_gems - rubies = 45 := by
  sorry

end diamonds_count_l2947_294787


namespace quadratic_inequality_solution_l2947_294781

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of the original inequality
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | (1/2 : ℝ) < x ∧ x < 2}

-- State the theorem
theorem quadratic_inequality_solution :
  ∃ (a : ℝ), 
    (∀ x, x ∈ solution_set a ↔ f a x > 0) ∧
    (a = -2) ∧
    (∀ x, a * x^2 - 5 * x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
sorry

end quadratic_inequality_solution_l2947_294781


namespace meetings_percentage_theorem_l2947_294734

/-- Calculates the percentage of a workday spent in meetings -/
def percentage_in_meetings (workday_hours : ℕ) (first_meeting_minutes : ℕ) (second_meeting_multiplier : ℕ) : ℚ :=
  let workday_minutes : ℕ := workday_hours * 60
  let second_meeting_minutes : ℕ := first_meeting_minutes * second_meeting_multiplier
  let total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes
  (total_meeting_minutes : ℚ) / (workday_minutes : ℚ) * 100

theorem meetings_percentage_theorem (workday_hours : ℕ) (first_meeting_minutes : ℕ) (second_meeting_multiplier : ℕ)
  (h1 : workday_hours = 10)
  (h2 : first_meeting_minutes = 60)
  (h3 : second_meeting_multiplier = 3) :
  percentage_in_meetings workday_hours first_meeting_minutes second_meeting_multiplier = 40 := by
  sorry

end meetings_percentage_theorem_l2947_294734


namespace B_equals_zero_one_two_l2947_294799

def A : Set ℤ := {1, 0, -1, 2}

def B : Set ℕ := {y | ∃ x ∈ A, y = |x|}

theorem B_equals_zero_one_two : B = {0, 1, 2} := by sorry

end B_equals_zero_one_two_l2947_294799


namespace at_hash_product_l2947_294780

-- Define the @ operation
def at_op (a b c : ℤ) : ℤ := a * b - b^2 + c

-- Define the # operation
def hash_op (a b c : ℤ) : ℤ := a + b - a * b^2 + c

-- Theorem statement
theorem at_hash_product : 
  let c : ℤ := 3
  (at_op 4 3 c) * (hash_op 4 3 c) = -156 := by
  sorry

end at_hash_product_l2947_294780


namespace square_of_real_not_always_positive_l2947_294725

theorem square_of_real_not_always_positive : 
  ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end square_of_real_not_always_positive_l2947_294725


namespace bicycle_sale_profit_l2947_294737

theorem bicycle_sale_profit (final_price : ℝ) (initial_cost : ℝ) (intermediate_profit_rate : ℝ) :
  final_price = 225 →
  initial_cost = 150 →
  intermediate_profit_rate = 0.25 →
  ((final_price / (1 + intermediate_profit_rate) - initial_cost) / initial_cost) * 100 = 20 := by
sorry

end bicycle_sale_profit_l2947_294737


namespace min_horse_pony_difference_l2947_294785

/-- Represents a ranch with horses and ponies -/
structure Ranch where
  horses : ℕ
  ponies : ℕ
  horseshoed_ponies : ℕ
  icelandic_horseshoed_ponies : ℕ

/-- Conditions for the ranch -/
def valid_ranch (r : Ranch) : Prop :=
  r.horses > r.ponies ∧
  r.horses + r.ponies = 164 ∧
  r.horseshoed_ponies = (3 * r.ponies) / 10 ∧
  r.icelandic_horseshoed_ponies = (5 * r.horseshoed_ponies) / 8

theorem min_horse_pony_difference (r : Ranch) (h : valid_ranch r) :
  r.horses - r.ponies = 4 := by
  sorry

end min_horse_pony_difference_l2947_294785


namespace octal_to_binary_conversion_l2947_294709

theorem octal_to_binary_conversion :
  (135 : Nat).digits 8 = [1, 3, 5] →
  (135 : Nat).digits 2 = [1, 0, 1, 1, 1, 0, 1] :=
by
  sorry

end octal_to_binary_conversion_l2947_294709


namespace beaver_dam_theorem_l2947_294703

/-- The number of hours it takes the first group of beavers to build the dam -/
def first_group_time : ℝ := 8

/-- The number of beavers in the second group -/
def second_group_size : ℝ := 36

/-- The number of hours it takes the second group of beavers to build the dam -/
def second_group_time : ℝ := 4

/-- The number of beavers in the first group -/
def first_group_size : ℝ := 18

theorem beaver_dam_theorem :
  first_group_size * first_group_time = second_group_size * second_group_time :=
by sorry

#check beaver_dam_theorem

end beaver_dam_theorem_l2947_294703
