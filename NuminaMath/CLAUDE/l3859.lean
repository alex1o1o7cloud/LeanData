import Mathlib

namespace NUMINAMATH_CALUDE_emily_age_proof_l3859_385921

/-- Rachel's current age -/
def rachel_current_age : ℕ := 24

/-- Rachel's age when Emily was half her age -/
def rachel_past_age : ℕ := 8

/-- Emily's age when Rachel was 8 -/
def emily_past_age : ℕ := rachel_past_age / 2

/-- The constant age difference between Rachel and Emily -/
def age_difference : ℕ := rachel_past_age - emily_past_age

/-- Emily's current age -/
def emily_current_age : ℕ := rachel_current_age - age_difference

theorem emily_age_proof : emily_current_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_age_proof_l3859_385921


namespace NUMINAMATH_CALUDE_power_product_squared_l3859_385989

theorem power_product_squared (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l3859_385989


namespace NUMINAMATH_CALUDE_larger_square_side_length_l3859_385913

theorem larger_square_side_length 
  (shaded_area unshaded_area : ℝ) 
  (h1 : shaded_area = 18)
  (h2 : unshaded_area = 18) : 
  ∃ (side_length : ℝ), side_length = 6 ∧ side_length^2 = shaded_area + unshaded_area :=
by
  sorry

end NUMINAMATH_CALUDE_larger_square_side_length_l3859_385913


namespace NUMINAMATH_CALUDE_metal_bar_weight_l3859_385998

/-- The weight of Harry's custom creation at the gym -/
def total_weight : ℕ := 25

/-- The weight of each blue weight -/
def blue_weight : ℕ := 2

/-- The weight of each green weight -/
def green_weight : ℕ := 3

/-- The number of blue weights Harry put on the bar -/
def num_blue_weights : ℕ := 4

/-- The number of green weights Harry put on the bar -/
def num_green_weights : ℕ := 5

/-- The weight of the metal bar -/
def bar_weight : ℕ := total_weight - (num_blue_weights * blue_weight + num_green_weights * green_weight)

theorem metal_bar_weight : bar_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_metal_bar_weight_l3859_385998


namespace NUMINAMATH_CALUDE_trig_identity_l3859_385960

theorem trig_identity (α : Real) (h : Real.sin α + Real.sin α ^ 2 = 1) :
  Real.cos α ^ 2 + Real.cos α ^ 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3859_385960


namespace NUMINAMATH_CALUDE_line_slope_l3859_385933

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 6 * x + 7 * y - 3 = 0

-- State the theorem
theorem line_slope :
  ∃ m b : ℝ, (∀ x y : ℝ, line_equation x y ↔ y = m * x + b) ∧ m = -6/7 :=
sorry

end NUMINAMATH_CALUDE_line_slope_l3859_385933


namespace NUMINAMATH_CALUDE_min_value_product_quotient_l3859_385930

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 3*x + 2) * (y^2 + 3*y + 2) * (z^2 + 3*z + 2) / (x*y*z) ≥ 216 ∧
  (x^2 + 3*x + 2) * (y^2 + 3*y + 2) * (z^2 + 3*z + 2) / (x*y*z) = 216 ↔ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_l3859_385930


namespace NUMINAMATH_CALUDE_payment_proof_l3859_385973

/-- Given a total payment of $80 using $20 and $10 bills, where the number of $20 bills
    is one more than the number of $10 bills, prove that the number of $10 bills used is 2. -/
theorem payment_proof (total : ℕ) (ten_bills : ℕ) (twenty_bills : ℕ) : 
  total = 80 →
  twenty_bills = ten_bills + 1 →
  10 * ten_bills + 20 * twenty_bills = total →
  ten_bills = 2 := by
  sorry

end NUMINAMATH_CALUDE_payment_proof_l3859_385973


namespace NUMINAMATH_CALUDE_minimum_dimes_for_scarf_l3859_385980

/-- The cost of the scarf in cents -/
def scarf_cost : ℕ := 4285

/-- The amount of money Chloe has without dimes, in cents -/
def initial_money : ℕ := 4000 + 100 + 50

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The minimum number of dimes needed to buy the scarf -/
def min_dimes_needed : ℕ := 14

theorem minimum_dimes_for_scarf :
  min_dimes_needed = (scarf_cost - initial_money + dime_value - 1) / dime_value :=
by sorry

end NUMINAMATH_CALUDE_minimum_dimes_for_scarf_l3859_385980


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3859_385920

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (2 * x - 3) = 10 → x = 103 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3859_385920


namespace NUMINAMATH_CALUDE_mobileRadiationNotSuitable_l3859_385940

/-- Represents a statistical activity that can be potentially collected through a questionnaire. -/
inductive StatisticalActivity
  | BlueCars
  | TVsInHomes
  | WakeUpTime
  | MobileRadiation

/-- Predicate to determine if a statistical activity is suitable for questionnaire data collection. -/
def suitableForQuestionnaire (activity : StatisticalActivity) : Prop :=
  match activity with
  | StatisticalActivity.BlueCars => True
  | StatisticalActivity.TVsInHomes => True
  | StatisticalActivity.WakeUpTime => True
  | StatisticalActivity.MobileRadiation => False

/-- Theorem stating that mobile radiation is the only activity not suitable for questionnaire data collection. -/
theorem mobileRadiationNotSuitable :
    ∀ (activity : StatisticalActivity),
      ¬(suitableForQuestionnaire activity) ↔ activity = StatisticalActivity.MobileRadiation := by
  sorry

end NUMINAMATH_CALUDE_mobileRadiationNotSuitable_l3859_385940


namespace NUMINAMATH_CALUDE_mike_mark_height_difference_l3859_385932

/-- Converts feet and inches to total inches -/
def height_to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- The height difference between two people in inches -/
def height_difference (height1 : ℕ) (height2 : ℕ) : ℕ := 
  if height1 ≥ height2 then height1 - height2 else height2 - height1

theorem mike_mark_height_difference :
  let mark_height := height_to_inches 5 3
  let mike_height := height_to_inches 6 1
  height_difference mike_height mark_height = 10 := by
sorry

end NUMINAMATH_CALUDE_mike_mark_height_difference_l3859_385932


namespace NUMINAMATH_CALUDE_rhombus_area_l3859_385929

/-- The area of a rhombus given its vertices in a rectangular coordinate system -/
theorem rhombus_area (A B C D : ℝ × ℝ) : 
  A = (2, 5.5) → 
  B = (8.5, 1) → 
  C = (2, -3.5) → 
  D = (-4.5, 1) → 
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BD : ℝ × ℝ := (D.1 - B.1, D.2 - B.2)
  let cross_product : ℝ := AC.1 * BD.2 - AC.2 * BD.1
  0.5 * |cross_product| = 58.5 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l3859_385929


namespace NUMINAMATH_CALUDE_min_flash_drives_l3859_385981

theorem min_flash_drives (total_files : ℕ) (drive_capacity : ℚ)
  (files_0_9MB : ℕ) (files_0_8MB : ℕ) (files_0_6MB : ℕ) :
  total_files = files_0_9MB + files_0_8MB + files_0_6MB →
  drive_capacity = 2.88 →
  files_0_9MB = 5 →
  files_0_8MB = 18 →
  files_0_6MB = 17 →
  (∃ min_drives : ℕ, 
    min_drives = 13 ∧
    min_drives * drive_capacity ≥ 
      (files_0_9MB * 0.9 + files_0_8MB * 0.8 + files_0_6MB * 0.6) ∧
    ∀ n : ℕ, n < min_drives → 
      n * drive_capacity < 
        (files_0_9MB * 0.9 + files_0_8MB * 0.8 + files_0_6MB * 0.6)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_flash_drives_l3859_385981


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3859_385901

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes
  (l m : Line) (α β : Plane)
  (different_lines : l ≠ m)
  (non_coincident_planes : α ≠ β)
  (l_perp_α : perpendicular l α)
  (α_parallel_β : parallel α β)
  (m_in_β : contained_in m β) :
  line_perpendicular l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3859_385901


namespace NUMINAMATH_CALUDE_condition_relationships_l3859_385912

theorem condition_relationships (α β γ : Prop) 
  (h1 : β → α)  -- α is necessary for β
  (h2 : ¬(α → β))  -- α is not sufficient for β
  (h3 : γ ↔ β)  -- γ is necessary and sufficient for β
  : (γ → α) ∧ ¬(α → γ) := by sorry

end NUMINAMATH_CALUDE_condition_relationships_l3859_385912


namespace NUMINAMATH_CALUDE_tissues_left_proof_l3859_385988

/-- The number of tissues left after buying boxes and using some tissues. -/
def tissues_left (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_used : ℕ) : ℕ :=
  tissues_per_box * boxes_bought - tissues_used

/-- Theorem: Given the conditions, prove that the number of tissues left is 270. -/
theorem tissues_left_proof :
  let tissues_per_box : ℕ := 160
  let boxes_bought : ℕ := 3
  let tissues_used : ℕ := 210
  tissues_left tissues_per_box boxes_bought tissues_used = 270 := by
  sorry

end NUMINAMATH_CALUDE_tissues_left_proof_l3859_385988


namespace NUMINAMATH_CALUDE_whiteboard_ink_cost_l3859_385977

/-- Calculates the cost of whiteboard ink usage for one day -/
theorem whiteboard_ink_cost (num_classes : ℕ) (boards_per_class : ℕ) (ink_per_board : ℝ) (cost_per_ml : ℝ) : 
  num_classes = 5 → 
  boards_per_class = 2 → 
  ink_per_board = 20 → 
  cost_per_ml = 0.5 → 
  (num_classes * boards_per_class * ink_per_board * cost_per_ml : ℝ) = 100 := by
sorry

end NUMINAMATH_CALUDE_whiteboard_ink_cost_l3859_385977


namespace NUMINAMATH_CALUDE_income_distribution_l3859_385975

theorem income_distribution (total_income : ℝ) (wife_percentage : ℝ) (orphan_percentage : ℝ) 
  (final_amount : ℝ) (num_children : ℕ) :
  total_income = 1000 →
  wife_percentage = 0.2 →
  orphan_percentage = 0.1 →
  final_amount = 500 →
  num_children = 2 →
  let remaining_after_wife := total_income * (1 - wife_percentage)
  let remaining_after_orphan := remaining_after_wife * (1 - orphan_percentage)
  let amount_to_children := remaining_after_orphan - final_amount
  let amount_per_child := amount_to_children / num_children
  amount_per_child / total_income = 0.11 := by
sorry

end NUMINAMATH_CALUDE_income_distribution_l3859_385975


namespace NUMINAMATH_CALUDE_hamburger_sales_l3859_385983

theorem hamburger_sales (total_target : ℕ) (price_per_hamburger : ℕ) (remaining_hamburgers : ℕ) : 
  total_target = 50 →
  price_per_hamburger = 5 →
  remaining_hamburgers = 4 →
  (total_target - remaining_hamburgers * price_per_hamburger) / price_per_hamburger = 6 :=
by sorry

end NUMINAMATH_CALUDE_hamburger_sales_l3859_385983


namespace NUMINAMATH_CALUDE_gift_distribution_sequences_l3859_385902

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of ways to distribute gifts in one class session -/
def ways_per_session : ℕ := num_students * num_students

/-- The total number of different gift distribution sequences in a week -/
def total_sequences : ℕ := ways_per_session ^ meetings_per_week

/-- Theorem stating the total number of different gift distribution sequences -/
theorem gift_distribution_sequences :
  total_sequences = 11390625 := by
  sorry

end NUMINAMATH_CALUDE_gift_distribution_sequences_l3859_385902


namespace NUMINAMATH_CALUDE_profit_percentage_is_20_l3859_385969

-- Define the quantities and prices
def wheat1_quantity : ℝ := 30
def wheat1_price : ℝ := 11.50
def wheat2_quantity : ℝ := 20
def wheat2_price : ℝ := 14.25
def selling_price : ℝ := 15.12

-- Define the theorem
theorem profit_percentage_is_20 : 
  let total_cost := wheat1_quantity * wheat1_price + wheat2_quantity * wheat2_price
  let total_weight := wheat1_quantity + wheat2_quantity
  let cost_price_per_kg := total_cost / total_weight
  let profit_per_kg := selling_price - cost_price_per_kg
  let profit_percentage := (profit_per_kg / cost_price_per_kg) * 100
  profit_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_is_20_l3859_385969


namespace NUMINAMATH_CALUDE_remainder_problem_l3859_385954

theorem remainder_problem (x : Int) : 
  x % 14 = 11 → x % 84 = 81 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l3859_385954


namespace NUMINAMATH_CALUDE_square_value_l3859_385931

theorem square_value : ∃ (square : ℚ), (7863 : ℚ) / 13 = 604 + square / 13 ∧ square = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l3859_385931


namespace NUMINAMATH_CALUDE_function_F_property_l3859_385946

-- Define the function F
noncomputable def F : ℝ → ℝ := sorry

-- State the theorem
theorem function_F_property (x : ℝ) : 
  (F ((1 - x) / (1 + x)) = x) → 
  (F (-2 - x) = -2 - F x) := by sorry

end NUMINAMATH_CALUDE_function_F_property_l3859_385946


namespace NUMINAMATH_CALUDE_pizza_fraction_eaten_l3859_385985

theorem pizza_fraction_eaten (total_slices : ℕ) (whole_slices_eaten : ℕ) (shared_slices : ℕ) :
  total_slices = 16 →
  whole_slices_eaten = 2 →
  shared_slices = 2 →
  (whole_slices_eaten : ℚ) / total_slices + (shared_slices : ℚ) / (2 * total_slices) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_eaten_l3859_385985


namespace NUMINAMATH_CALUDE_locus_of_midpoints_l3859_385958

-- Define the circle L
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a point Q inside the circle
def Q (L : Circle) : ℝ × ℝ :=
  sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

-- State the theorem
theorem locus_of_midpoints (L : Circle) :
  -- Q is an interior point of L
  (distance (Q L) L.center < L.radius) →
  -- Q is not the center of L
  (Q L ≠ L.center) →
  -- The distance from Q to the center of L is one-third the radius of L
  (distance (Q L) L.center = L.radius / 3) →
  -- The locus of midpoints of all chords passing through Q is a complete circle
  ∃ (C : Circle),
    -- The center of the locus circle is Q
    C.center = Q L ∧
    -- The radius of the locus circle is r/6
    C.radius = L.radius / 6 :=
sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_l3859_385958


namespace NUMINAMATH_CALUDE_definite_integral_ln_squared_over_sqrt_l3859_385992

theorem definite_integral_ln_squared_over_sqrt (e : Real) :
  let f : Real → Real := fun x => (Real.log x)^2 / Real.sqrt x
  let a : Real := 1
  let b : Real := Real.exp 2
  e > 0 →
  ∫ x in a..b, f x = 24 * e - 32 := by
sorry

end NUMINAMATH_CALUDE_definite_integral_ln_squared_over_sqrt_l3859_385992


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l3859_385937

theorem largest_number_in_ratio (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (b = (5 * a) / 3) → 
  (c = (7 * a) / 3) → 
  (c - a = 40) → 
  c = 70 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l3859_385937


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3859_385941

theorem quadratic_root_difference : 
  let a : ℝ := 5 + 3 * Real.sqrt 5
  let b : ℝ := 5 + Real.sqrt 5
  let c : ℝ := -3
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  abs (root1 - root2) = 1/2 + 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3859_385941


namespace NUMINAMATH_CALUDE_selection_methods_with_female_l3859_385916

def total_students : ℕ := 8
def male_students : ℕ := 4
def female_students : ℕ := 4
def students_to_select : ℕ := 3

theorem selection_methods_with_female (h1 : total_students = male_students + female_students) 
  (h2 : total_students ≥ students_to_select) :
  (Nat.choose total_students students_to_select) - (Nat.choose male_students students_to_select) = 52 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_with_female_l3859_385916


namespace NUMINAMATH_CALUDE_carousel_seating_arrangement_l3859_385990

-- Define the friends
inductive Friend
| Alan
| Bella
| Chloe
| David
| Emma

-- Define the seats
inductive Seat
| One
| Two
| Three
| Four
| Five

-- Define the seating arrangement
def SeatingArrangement := Friend → Seat

-- Define the condition of being opposite
def isOpposite (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ s2 = Seat.Three) ∨ (s1 = Seat.Two ∧ s2 = Seat.Four) ∨
  (s1 = Seat.Three ∧ s2 = Seat.Five) ∨ (s1 = Seat.Four ∧ s2 = Seat.One) ∨
  (s1 = Seat.Five ∧ s2 = Seat.Two)

-- Define the condition of being two seats away
def isTwoSeatsAway (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ s2 = Seat.Three) ∨ (s1 = Seat.Two ∧ s2 = Seat.Four) ∨
  (s1 = Seat.Three ∧ s2 = Seat.Five) ∨ (s1 = Seat.Four ∧ s2 = Seat.One) ∨
  (s1 = Seat.Five ∧ s2 = Seat.Two)

-- Define the condition of being next to each other
def isNextTo (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ (s2 = Seat.Two ∨ s2 = Seat.Five)) ∨
  (s1 = Seat.Two ∧ (s2 = Seat.One ∨ s2 = Seat.Three)) ∨
  (s1 = Seat.Three ∧ (s2 = Seat.Two ∨ s2 = Seat.Four)) ∨
  (s1 = Seat.Four ∧ (s2 = Seat.Three ∨ s2 = Seat.Five)) ∨
  (s1 = Seat.Five ∧ (s2 = Seat.Four ∨ s2 = Seat.One))

-- Define the condition of being to the immediate left
def isImmediateLeft (s1 s2 : Seat) : Prop :=
  (s1 = Seat.One ∧ s2 = Seat.Two) ∨
  (s1 = Seat.Two ∧ s2 = Seat.Three) ∨
  (s1 = Seat.Three ∧ s2 = Seat.Four) ∨
  (s1 = Seat.Four ∧ s2 = Seat.Five) ∨
  (s1 = Seat.Five ∧ s2 = Seat.One)

theorem carousel_seating_arrangement 
  (seating : SeatingArrangement)
  (h1 : isOpposite (seating Friend.Chloe) (seating Friend.Emma))
  (h2 : isTwoSeatsAway (seating Friend.David) (seating Friend.Alan))
  (h3 : ¬isNextTo (seating Friend.Alan) (seating Friend.Emma))
  (h4 : isNextTo (seating Friend.Bella) (seating Friend.Emma))
  : isImmediateLeft (seating Friend.Chloe) (seating Friend.Alan) :=
sorry

end NUMINAMATH_CALUDE_carousel_seating_arrangement_l3859_385990


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l3859_385945

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for numbers using only specified digits -/
def uses_specified_digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_M (M : ℕ) 
  (h_even : Even M)
  (h_digits : uses_specified_digits M)
  (h_double : sum_of_digits (2 * M) = 39)
  (h_half : sum_of_digits (M / 2) = 30) :
  sum_of_digits M = 33 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l3859_385945


namespace NUMINAMATH_CALUDE_quadratic_equation_identification_l3859_385900

/-- Definition of a quadratic equation in one variable -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equations given in the problem -/
def eq_A : ℝ → ℝ := λ x => 2 * x - 1
def eq_B : ℝ → ℝ := λ x => x^2
def eq_C : ℝ → ℝ → ℝ := λ x y => 5 * x * y - 1
def eq_D : ℝ → ℝ := λ x => 2 * (x + 1)

/-- Theorem stating that eq_B is quadratic while others are not -/
theorem quadratic_equation_identification :
  is_quadratic eq_B ∧ 
  ¬is_quadratic eq_A ∧ 
  ¬is_quadratic (λ x => eq_C x x) ∧ 
  ¬is_quadratic eq_D :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_identification_l3859_385900


namespace NUMINAMATH_CALUDE_player_A_wins_l3859_385923

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents the state of the blackboard -/
def BoardState : Type := List Nat

/-- Checks if a move is valid according to the game rules -/
def isValidMove (current : BoardState) (next : BoardState) : Prop :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy : Type := BoardState → BoardState

/-- Checks if a strategy is winning for a given player -/
def isWinningStrategy (player : Player) (strat : Strategy) : Prop :=
  sorry

/-- The initial state of the board -/
def initialState : BoardState := [10^2007]

/-- The main theorem stating that Player A has a winning strategy -/
theorem player_A_wins :
  ∃ (strat : Strategy), isWinningStrategy Player.A strat :=
sorry

end NUMINAMATH_CALUDE_player_A_wins_l3859_385923


namespace NUMINAMATH_CALUDE_equation_solution_l3859_385908

theorem equation_solution : ∃! x : ℝ, (Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 4) ∧ (x = 4 * Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3859_385908


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3859_385936

def A : Set ℕ := {x | ∃ n : ℕ, x = 3 * n + 2}
def B : Set ℕ := {6, 8, 10, 12, 14}

theorem intersection_of_A_and_B : A ∩ B = {8, 14} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3859_385936


namespace NUMINAMATH_CALUDE_excursion_min_parents_l3859_385907

/-- The minimum number of parents needed for an excursion -/
def min_parents_needed (num_students : ℕ) (car_capacity : ℕ) : ℕ :=
  Nat.ceil (num_students / (car_capacity - 1))

/-- Theorem: The minimum number of parents needed for 30 students with 5-seat cars is 8 -/
theorem excursion_min_parents :
  min_parents_needed 30 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_excursion_min_parents_l3859_385907


namespace NUMINAMATH_CALUDE_number_of_divisors_of_M_l3859_385926

def M : ℕ := 2^6 * 3^4 * 5^2 * 7^2 * 11^1

theorem number_of_divisors_of_M : (Nat.divisors M).card = 630 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_M_l3859_385926


namespace NUMINAMATH_CALUDE_inequality_proof_l3859_385972

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 4) 
  (h2 : c^2 + d^2 = 16) : 
  a*c + b*d ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3859_385972


namespace NUMINAMATH_CALUDE_postage_cost_correct_l3859_385956

-- Define the postage pricing structure
def base_rate : ℚ := 50 / 100
def additional_rate : ℚ := 15 / 100
def weight_increment : ℚ := 1 / 2
def package_weight : ℚ := 28 / 10
def cost_cap : ℚ := 130 / 100

-- Calculate the postage cost
def postage_cost : ℚ :=
  base_rate + additional_rate * (Int.ceil ((package_weight - 1) / weight_increment))

-- Theorem to prove
theorem postage_cost_correct : 
  postage_cost = 110 / 100 ∧ postage_cost ≤ cost_cap := by
  sorry

end NUMINAMATH_CALUDE_postage_cost_correct_l3859_385956


namespace NUMINAMATH_CALUDE_value_of_expression_l3859_385978

theorem value_of_expression (m n : ℤ) (h1 : |m| = 5) (h2 : |n| = 4) (h3 : m * n < 0) :
  m^2 - m*n + n = 41 ∨ m^2 - m*n + n = 49 :=
sorry

end NUMINAMATH_CALUDE_value_of_expression_l3859_385978


namespace NUMINAMATH_CALUDE_missing_sale_is_8562_l3859_385942

/-- Calculates the missing sale amount given sales for 5 months and the average -/
def calculate_missing_sale (sale1 sale2 sale3 sale4 sale6 average : ℚ) : ℚ :=
  6 * average - (sale1 + sale2 + sale3 + sale4 + sale6)

theorem missing_sale_is_8562 :
  let sale1 : ℚ := 8435
  let sale2 : ℚ := 8927
  let sale3 : ℚ := 8855
  let sale4 : ℚ := 9230
  let sale6 : ℚ := 6991
  let average : ℚ := 8500
  calculate_missing_sale sale1 sale2 sale3 sale4 sale6 average = 8562 := by
  sorry

#eval calculate_missing_sale 8435 8927 8855 9230 6991 8500

end NUMINAMATH_CALUDE_missing_sale_is_8562_l3859_385942


namespace NUMINAMATH_CALUDE_salesman_pears_morning_sales_salesman_pears_morning_sales_proof_l3859_385966

/-- Proof that a salesman sold 120 kilograms of pears in the morning -/
theorem salesman_pears_morning_sales : ℝ → Prop :=
  fun morning_sales : ℝ =>
    let afternoon_sales := 240
    let total_sales := 360
    (afternoon_sales = 2 * morning_sales) ∧
    (total_sales = morning_sales + afternoon_sales) →
    morning_sales = 120

-- The proof is omitted
theorem salesman_pears_morning_sales_proof : salesman_pears_morning_sales 120 := by
  sorry

end NUMINAMATH_CALUDE_salesman_pears_morning_sales_salesman_pears_morning_sales_proof_l3859_385966


namespace NUMINAMATH_CALUDE_time_addition_theorem_l3859_385938

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time -/
def addTime (initial : Time) (dHours dMinutes dSeconds : Nat) : Time :=
  sorry

/-- Converts 24-hour time to 12-hour time -/
def to12Hour (time : Time) : Time :=
  sorry

/-- Calculates the sum of hours, minutes, and seconds -/
def sumTimeComponents (time : Time) : Nat :=
  sorry

theorem time_addition_theorem :
  let initial_time := Time.mk 15 15 30  -- 3:15:30 PM
  let duration_hours := 174
  let duration_minutes := 58
  let duration_seconds := 16
  let final_time := to12Hour (addTime initial_time duration_hours duration_minutes duration_seconds)
  final_time = Time.mk 10 13 46 ∧ sumTimeComponents final_time = 69 := by
  sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l3859_385938


namespace NUMINAMATH_CALUDE_parabola_equation_l3859_385995

/-- A parabola with vertex at the origin, coordinate axes as axes of symmetry, 
    and passing through point (-4, -2) has a standard equation of either 
    x^2 = -8y or y^2 = -x -/
theorem parabola_equation (f : ℝ → ℝ) : 
  (∀ x y, f x = y ↔ (x^2 = -8*y ∨ y^2 = -x)) ↔ 
  (f 0 = 0 ∧ 
   (∀ x, f x = f (-x)) ∧ 
   (∀ y, f (f y) = y) ∧
   f (-4) = -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3859_385995


namespace NUMINAMATH_CALUDE_masha_numbers_proof_l3859_385952

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a % 2 = 0 ∨ b % 2 = 0)

def is_unique_pair (a b : ℕ) : Prop :=
  ∀ x y : ℕ, x + y = a + b → is_valid_pair x y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)

theorem masha_numbers_proof :
  ∃! (a b : ℕ), is_valid_pair a b ∧ is_unique_pair a b ∧ a + b = 28 :=
sorry

end NUMINAMATH_CALUDE_masha_numbers_proof_l3859_385952


namespace NUMINAMATH_CALUDE_odd_sum_ends_with_1379_l3859_385927

theorem odd_sum_ends_with_1379 (S : Finset ℕ) 
  (h1 : S.card = 10000)
  (h2 : ∀ n ∈ S, Odd n)
  (h3 : ∀ n ∈ S, ¬(5 ∣ n)) :
  ∃ T ⊆ S, (T.sum id) % 10000 = 1379 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_ends_with_1379_l3859_385927


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3859_385919

theorem tan_alpha_value (α : Real) (h : Real.cos α + 2 * Real.sin α = Real.sqrt 5) :
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3859_385919


namespace NUMINAMATH_CALUDE_log_equality_implies_y_value_l3859_385950

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the theorem
theorem log_equality_implies_y_value 
  (a b c x : ℝ) 
  (p q r y : ℝ) 
  (h1 : log a / p = log b / q)
  (h2 : log b / q = log c / r)
  (h3 : log c / r = log x)
  (h4 : x ≠ 1)
  (h5 : b^3 / (a^2 * c) = x^y) :
  y = 3*q - 2*p - r := by
  sorry

#check log_equality_implies_y_value

end NUMINAMATH_CALUDE_log_equality_implies_y_value_l3859_385950


namespace NUMINAMATH_CALUDE_fixed_distance_theorem_l3859_385939

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_fixed_distance (p a b : E) : Prop :=
  ∃ (c : ℝ), ∀ (q : E), ‖p - b‖ = 3 * ‖p - a‖ → ‖q - b‖ = 3 * ‖q - a‖ → 
    ‖p - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖ = ‖q - ((9/8 : ℝ) • a - (1/8 : ℝ) • b)‖

theorem fixed_distance_theorem (a b p : E) :
  ‖p - b‖ = 3 * ‖p - a‖ → is_fixed_distance p a b :=
by sorry

end NUMINAMATH_CALUDE_fixed_distance_theorem_l3859_385939


namespace NUMINAMATH_CALUDE_smallest_cube_root_integer_l3859_385962

theorem smallest_cube_root_integer (m n : ℕ) (s : ℝ) : 
  (0 < n) →
  (0 < s) →
  (s < 1 / 2000) →
  (m = (n + s)^3) →
  (∀ k < n, ∀ t > 0, t < 1 / 2000 → ¬ (∃ l : ℕ, l = (k + t)^3)) →
  (n = 26) := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_integer_l3859_385962


namespace NUMINAMATH_CALUDE_sum_of_arcs_equals_180_degrees_l3859_385994

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define an arc on a circle
structure Arc :=
  (circle : Circle)
  (start_angle : ℝ)
  (end_angle : ℝ)

-- Define the arrangement of three circles
def triangle_arrangement (c1 c2 c3 : Circle) : Prop :=
  -- This is a placeholder for the specific arrangement condition
  True

-- Define the theorem
theorem sum_of_arcs_equals_180_degrees 
  (c1 c2 c3 : Circle) 
  (ab : Arc) 
  (cd : Arc) 
  (ef : Arc) 
  (h1 : c1.radius = c2.radius ∧ c2.radius = c3.radius)
  (h2 : triangle_arrangement c1 c2 c3)
  (h3 : ab.circle = c1 ∧ cd.circle = c2 ∧ ef.circle = c3) :
  ab.end_angle - ab.start_angle + 
  cd.end_angle - cd.start_angle + 
  ef.end_angle - ef.start_angle = π :=
sorry

end NUMINAMATH_CALUDE_sum_of_arcs_equals_180_degrees_l3859_385994


namespace NUMINAMATH_CALUDE_cone_base_radius_l3859_385943

/-- Given a cone with slant height 12 cm and central angle of unfolded lateral surface 150°, 
    the radius of its base is 5 cm. -/
theorem cone_base_radius (slant_height : ℝ) (central_angle : ℝ) : 
  slant_height = 12 → central_angle = 150 → ∃ (base_radius : ℝ), base_radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3859_385943


namespace NUMINAMATH_CALUDE_ninth_root_unity_sum_l3859_385914

theorem ninth_root_unity_sum (z : ℂ) : 
  z = Complex.exp (Complex.I * (2 * Real.pi / 9)) →
  z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^6 / (1 + z^9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ninth_root_unity_sum_l3859_385914


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3859_385986

theorem polynomial_coefficient_sum (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₁ * (x - 1)^4 + a₂ * (x - 1)^3 + a₃ * (x - 1)^2 + a₄ * (x - 1) + a₅ = x^4) →
  a₂ - a₃ + a₄ = 2 := by
sorry


end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3859_385986


namespace NUMINAMATH_CALUDE_exists_iceberg_with_properties_l3859_385935

/-- Represents a convex polyhedron floating in water --/
structure FloatingPolyhedron where
  totalVolume : ℝ
  submergedVolume : ℝ
  totalSurfaceArea : ℝ
  submergedSurfaceArea : ℝ
  volume_nonneg : 0 < totalVolume
  submerged_volume_le_total : submergedVolume ≤ totalVolume
  surface_area_nonneg : 0 < totalSurfaceArea
  submerged_surface_le_total : submergedSurfaceArea ≤ totalSurfaceArea

/-- Theorem stating the existence of a floating polyhedron with the required properties --/
theorem exists_iceberg_with_properties :
  ∃ (iceberg : FloatingPolyhedron),
    iceberg.submergedVolume ≥ 0.9 * iceberg.totalVolume ∧
    iceberg.submergedSurfaceArea ≤ 0.5 * iceberg.totalSurfaceArea :=
sorry

end NUMINAMATH_CALUDE_exists_iceberg_with_properties_l3859_385935


namespace NUMINAMATH_CALUDE_range_of_a_given_negative_root_l3859_385957

/-- Given that the equation 5^x = (a+3)/(5-a) has a negative root, 
    prove that the range of values for a is -3 < a < 1 -/
theorem range_of_a_given_negative_root (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ 5^x = (a+3)/(5-a)) → -3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_given_negative_root_l3859_385957


namespace NUMINAMATH_CALUDE_horner_method_operations_count_l3859_385970

def horner_polynomial (x : ℝ) : ℝ := 9*x^6 + 12*x^5 + 7*x^4 + 54*x^3 + 34*x^2 + 9*x + 1

def horner_method_operations (p : ℝ → ℝ) : ℕ × ℕ :=
  match p with
  | f => (6, 6)  -- Placeholder for the actual implementation

theorem horner_method_operations_count :
  ∀ x : ℝ, horner_method_operations horner_polynomial = (6, 6) := by
  sorry

end NUMINAMATH_CALUDE_horner_method_operations_count_l3859_385970


namespace NUMINAMATH_CALUDE_line_intersection_point_sum_l3859_385906

/-- The line equation y = -1/2x + 8 -/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (16, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 8)

/-- Point T is on the line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs (P.1 * Q.2 - Q.1 * P.2) / 2 = 4 * abs (r * P.2 - P.1 * s) / 2

theorem line_intersection_point_sum : 
  ∀ r s : ℝ, line_equation r s → T_on_PQ r s → area_condition r s → r + s = 14 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_point_sum_l3859_385906


namespace NUMINAMATH_CALUDE_article_cost_price_l3859_385967

theorem article_cost_price (C : ℝ) (S : ℝ) : 
  S = 1.05 * C ∧ 
  S - 1 = 1.045 * C → 
  C = 200 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l3859_385967


namespace NUMINAMATH_CALUDE_function_value_at_pi_third_l3859_385961

/-- Given a function f(x) = 2tan(ωx + φ) with the following properties:
    - ω > 0
    - |φ| < π/2
    - f(0) = 2√3/3
    - The period T ∈ (π/4, 3π/4)
    - (π/6, 0) is the center of symmetry of f(x)
    Prove that f(π/3) = -2√3/3 -/
theorem function_value_at_pi_third 
  (f : ℝ → ℝ) 
  (ω φ : ℝ) 
  (h1 : ∀ x, f x = 2 * Real.tan (ω * x + φ))
  (h2 : ω > 0)
  (h3 : abs φ < Real.pi / 2)
  (h4 : f 0 = 2 * Real.sqrt 3 / 3)
  (h5 : ∃ T, T ∈ Set.Ioo (Real.pi / 4) (3 * Real.pi / 4) ∧ ∀ x, f (x + T) = f x)
  (h6 : ∀ x, f (Real.pi / 3 - x) = f (Real.pi / 3 + x)) :
  f (Real.pi / 3) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_third_l3859_385961


namespace NUMINAMATH_CALUDE_farm_dogs_count_l3859_385949

theorem farm_dogs_count (num_houses : ℕ) (dogs_per_house : ℕ) (h1 : num_houses = 5) (h2 : dogs_per_house = 4) :
  num_houses * dogs_per_house = 20 := by
  sorry

end NUMINAMATH_CALUDE_farm_dogs_count_l3859_385949


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3859_385951

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3859_385951


namespace NUMINAMATH_CALUDE_value_of_expression_l3859_385982

theorem value_of_expression (s t : ℝ) 
  (hs : 19 * s^2 + 99 * s + 1 = 0)
  (ht : t^2 + 99 * t + 19 = 0)
  (hst : s * t ≠ 1) :
  (s * t + 4 * s + 1) / t = -5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3859_385982


namespace NUMINAMATH_CALUDE_unique_prime_with_no_cubic_sum_l3859_385963

-- Define the property for a prime p
def has_no_cubic_sum (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℤ, ∀ x y : ℤ, (x^3 + y^3) % p ≠ n % p

-- State the theorem
theorem unique_prime_with_no_cubic_sum :
  ∀ p : ℕ, has_no_cubic_sum p ↔ p = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_with_no_cubic_sum_l3859_385963


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3859_385928

theorem expand_and_simplify (x : ℝ) : 
  (1 + x^3) * (1 - x^4)^2 = 1 + x^3 - 2*x^4 - 2*x^7 + x^8 + x^11 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3859_385928


namespace NUMINAMATH_CALUDE_complex_number_modulus_l3859_385905

/-- Given a complex number z = (3ai)/(1-2i) where a < 0 and i is the imaginary unit,
    if |z| = √5, then a = -5/3 -/
theorem complex_number_modulus (a : ℝ) (h1 : a < 0) :
  let z : ℂ := (3 * a * Complex.I) / (1 - 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 → a = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l3859_385905


namespace NUMINAMATH_CALUDE_brothers_age_proof_l3859_385959

def hannah_age : ℕ := 48
def num_brothers : ℕ := 3

theorem brothers_age_proof (brothers_age : ℕ) 
  (h1 : hannah_age = 2 * (num_brothers * brothers_age)) : 
  brothers_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_proof_l3859_385959


namespace NUMINAMATH_CALUDE_oil_redistribution_l3859_385965

theorem oil_redistribution (trucks_type1 trucks_type2 boxes_per_truck1 boxes_per_truck2 containers_per_box final_trucks : ℕ) 
  (h1 : trucks_type1 = 7)
  (h2 : trucks_type2 = 5)
  (h3 : boxes_per_truck1 = 20)
  (h4 : boxes_per_truck2 = 12)
  (h5 : containers_per_box = 8)
  (h6 : final_trucks = 10) :
  (trucks_type1 * boxes_per_truck1 + trucks_type2 * boxes_per_truck2) * containers_per_box / final_trucks = 160 := by
  sorry

#check oil_redistribution

end NUMINAMATH_CALUDE_oil_redistribution_l3859_385965


namespace NUMINAMATH_CALUDE_sara_popsicle_consumption_l3859_385910

/-- The number of Popsicles Sara can eat in a given time period -/
def popsicles_eaten (minutes_per_popsicle : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / minutes_per_popsicle

theorem sara_popsicle_consumption :
  popsicles_eaten 20 340 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sara_popsicle_consumption_l3859_385910


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l3859_385991

theorem log_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log a / Real.log 9 = Real.log b / Real.log 12 ∧ 
       Real.log a / Real.log 9 = Real.log (a + b) / Real.log 16) : 
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l3859_385991


namespace NUMINAMATH_CALUDE_max_bug_contacts_l3859_385971

/-- The number of bugs on the stick -/
def total_bugs : ℕ := 2016

/-- The maximum number of contacts between bugs -/
def max_contacts : ℕ := 1016064

/-- Theorem stating that the maximum number of contacts is achieved when half the bugs move in each direction -/
theorem max_bug_contacts :
  ∀ (a b : ℕ), a + b = total_bugs → a * b ≤ max_contacts :=
by sorry

end NUMINAMATH_CALUDE_max_bug_contacts_l3859_385971


namespace NUMINAMATH_CALUDE_solution_set_l3859_385911

def f (x : ℝ) := 3 - 2*x

theorem solution_set (x : ℝ) : 
  x ∈ Set.Icc 0 3 ↔ |f (x + 1) + 2| ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3859_385911


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3859_385934

theorem solution_set_of_inequality (x : ℝ) :
  x * |x - 1| > 0 ↔ x ∈ Set.Ioo 0 1 ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3859_385934


namespace NUMINAMATH_CALUDE_min_value_expression_l3859_385993

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (2 * x^2 + 4 * y^2)).sqrt) / (x * y) ≥ 4 + 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3859_385993


namespace NUMINAMATH_CALUDE_total_popsicles_l3859_385925

theorem total_popsicles (grape : ℕ) (cherry : ℕ) (banana : ℕ) 
  (h1 : grape = 2) (h2 : cherry = 13) (h3 : banana = 2) : 
  grape + cherry + banana = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_popsicles_l3859_385925


namespace NUMINAMATH_CALUDE_max_sin_sum_l3859_385979

theorem max_sin_sum (α β θ : Real) : 
  α + β = 2 * Real.pi / 3 →
  α > 0 →
  β > 0 →
  (∀ x y, x + y = 2 * Real.pi / 3 → x > 0 → y > 0 → 
    Real.sin α + 2 * Real.sin β ≥ Real.sin x + 2 * Real.sin y) →
  α = θ →
  Real.cos θ = Real.sqrt 21 / 7 := by
sorry

end NUMINAMATH_CALUDE_max_sin_sum_l3859_385979


namespace NUMINAMATH_CALUDE_total_bulbs_is_469_l3859_385909

/-- Represents the number of lights of each type -/
structure LightCounts where
  tiny : ℕ
  small : ℕ
  medium : ℕ
  large : ℕ
  extraLarge : ℕ

/-- Calculates the total number of bulbs needed -/
def totalBulbs (counts : LightCounts) : ℕ :=
  counts.tiny * 1 + counts.small * 2 + counts.medium * 3 + counts.large * 4 + counts.extraLarge * 5

theorem total_bulbs_is_469 (counts : LightCounts) :
  counts.large = 2 * counts.medium →
  counts.small = (5 * counts.medium) / 4 →
  counts.extraLarge = counts.small - counts.tiny →
  4 * counts.tiny = 3 * counts.medium →
  2 * counts.small + 3 * counts.medium = 4 * counts.large + 5 * counts.extraLarge →
  counts.extraLarge = 14 →
  totalBulbs counts = 469 := by
  sorry

#eval totalBulbs { tiny := 21, small := 35, medium := 28, large := 56, extraLarge := 14 }

end NUMINAMATH_CALUDE_total_bulbs_is_469_l3859_385909


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3859_385999

theorem purely_imaginary_complex_number (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 + x - 2) (x + 2)
  (z.re = 0 ∧ z.im ≠ 0) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3859_385999


namespace NUMINAMATH_CALUDE_cost_of_one_plank_l3859_385997

/-- The cost of one plank given the conditions for building birdhouses -/
theorem cost_of_one_plank : 
  ∀ (plank_cost : ℝ),
  (4 * (7 * plank_cost + 20 * 0.05) = 88) →
  plank_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_plank_l3859_385997


namespace NUMINAMATH_CALUDE_sphere_surface_area_tangent_to_cube_l3859_385903

/-- The surface area of a sphere tangent to all faces of a cube -/
theorem sphere_surface_area_tangent_to_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) :
  cube_edge_length = 2 →
  sphere_radius = cube_edge_length / 2 →
  4 * Real.pi * sphere_radius^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_tangent_to_cube_l3859_385903


namespace NUMINAMATH_CALUDE_max_not_joined_company_l3859_385924

/-- The maximum number of people who did not join any club -/
def max_not_joined (total : ℕ) (m s z : ℕ) : ℕ :=
  total - (m + max s z)

/-- Proof that the maximum number of people who did not join any club is 26 -/
theorem max_not_joined_company : max_not_joined 60 16 18 11 = 26 := by
  sorry

end NUMINAMATH_CALUDE_max_not_joined_company_l3859_385924


namespace NUMINAMATH_CALUDE_car_trip_speed_l3859_385976

/-- Given a 6-hour trip with an average speed of 38 miles per hour,
    where the speed for the last 2 hours is 44 miles per hour,
    prove that the average speed for the first 4 hours is 35 miles per hour. -/
theorem car_trip_speed :
  ∀ (first_4_hours_speed : ℝ),
    (first_4_hours_speed * 4 + 44 * 2) / 6 = 38 →
    first_4_hours_speed = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_car_trip_speed_l3859_385976


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3859_385918

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations for parallel and perpendicular
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3859_385918


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l3859_385984

theorem smallest_four_digit_multiple_of_112 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 112 ∣ n → 1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l3859_385984


namespace NUMINAMATH_CALUDE_unique_solution_system_l3859_385904

/-- The system of equations has a unique solution when a = 5/3 and no solutions otherwise -/
theorem unique_solution_system (a x y : ℝ) : 
  (3 * (x - a)^2 + y = 2 - a) ∧ 
  (y^2 + ((x - 2) / (|x| - 2))^2 = 1) ∧ 
  (x ≥ 0) ∧ 
  (x ≠ 2) ↔ 
  (a = 5/3 ∧ x = 4/3 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3859_385904


namespace NUMINAMATH_CALUDE_number_at_21_21_l3859_385996

/-- Represents the number at a given position in the matrix -/
def matrixNumber (row : ℕ) (col : ℕ) : ℕ :=
  row^2 - (col - 1)

/-- The theorem stating that the number in the 21st row and 21st column is 421 -/
theorem number_at_21_21 : matrixNumber 21 21 = 421 := by
  sorry

end NUMINAMATH_CALUDE_number_at_21_21_l3859_385996


namespace NUMINAMATH_CALUDE_termite_ridden_collapsing_homes_l3859_385964

theorem termite_ridden_collapsing_homes 
  (total_homes : ℕ) 
  (termite_ridden : ℚ) 
  (termite_not_collapsing : ℚ) 
  (h1 : termite_ridden = 1/3) 
  (h2 : termite_not_collapsing = 1/7) : 
  (termite_ridden - termite_not_collapsing) / termite_ridden = 4/21 := by
sorry

end NUMINAMATH_CALUDE_termite_ridden_collapsing_homes_l3859_385964


namespace NUMINAMATH_CALUDE_meaningful_expression_l3859_385974

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (Real.sqrt (a + 1)) / (a - 2)) ↔ (a ≥ -1 ∧ a ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3859_385974


namespace NUMINAMATH_CALUDE_quadratic_function_range_difference_l3859_385968

-- Define the quadratic function
def f (x c : ℝ) : ℝ := -2 * x^2 + c

-- Define the theorem
theorem quadratic_function_range_difference (c m : ℝ) :
  (m + 2 ≤ 0) →
  (∃ (min : ℝ), ∀ (m' : ℝ), m' + 2 ≤ 0 → 
    (f (m' + 2) c - f m' c) ≥ min) ∧
  (¬∃ (max : ℝ), ∀ (m' : ℝ), m' + 2 ≤ 0 → 
    (f (m' + 2) c - f m' c) ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_difference_l3859_385968


namespace NUMINAMATH_CALUDE_arun_speed_ratio_l3859_385948

/-- Represents the problem of finding the ratio of Arun's new speed to his original speed. -/
theorem arun_speed_ratio :
  let distance : ℝ := 30
  let arun_original_speed : ℝ := 5
  let anil_time := distance / anil_speed
  let arun_original_time := distance / arun_original_speed
  let arun_new_time := distance / arun_new_speed
  arun_original_time = anil_time + 2 →
  arun_new_time = anil_time - 1 →
  arun_new_speed / arun_original_speed = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_arun_speed_ratio_l3859_385948


namespace NUMINAMATH_CALUDE_scientific_notation_929000_l3859_385917

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_929000 :
  scientific_notation 929000 = (9.29, 5) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_929000_l3859_385917


namespace NUMINAMATH_CALUDE_ring_diameter_theorem_l3859_385947

/-- The diameter of ring X -/
def diameter_X : ℝ := 16

/-- The fraction of ring X's surface not covered by ring Y -/
def uncovered_fraction : ℝ := 0.2098765432098765

/-- The diameter of ring Y -/
noncomputable def diameter_Y : ℝ := 14.222

/-- Theorem stating that given the diameter of ring X and the uncovered fraction,
    the diameter of ring Y is approximately 14.222 inches -/
theorem ring_diameter_theorem (ε : ℝ) (h : ε > 0) :
  ∃ (d : ℝ), abs (d - diameter_Y) < ε ∧ 
  d^2 / 4 = diameter_X^2 / 4 * (1 - uncovered_fraction) :=
sorry

end NUMINAMATH_CALUDE_ring_diameter_theorem_l3859_385947


namespace NUMINAMATH_CALUDE_new_average_after_drop_l3859_385955

/-- Theorem: New average after student drops class -/
theorem new_average_after_drop (n : ℕ) (old_avg : ℚ) (drop_score : ℚ) :
  n = 16 →
  old_avg = 62.5 →
  drop_score = 70 →
  (n : ℚ) * old_avg - drop_score = ((n - 1) : ℚ) * 62 :=
by sorry

end NUMINAMATH_CALUDE_new_average_after_drop_l3859_385955


namespace NUMINAMATH_CALUDE_satellite_sensor_ratio_l3859_385953

theorem satellite_sensor_ratio (total_units : Nat) (upgrade_fraction : Rat) : 
  total_units = 24 → 
  upgrade_fraction = 1 / 7 → 
  (∃ (non_upgraded_per_unit total_upgraded : Nat), 
    (non_upgraded_per_unit : Rat) / (total_upgraded : Rat) = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_satellite_sensor_ratio_l3859_385953


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3859_385922

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
    (h_a1 : a 1 = 8) (h_a4 : a 4 = 64) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3859_385922


namespace NUMINAMATH_CALUDE_geometric_sequence_m_value_l3859_385915

/-- Definition of the sum of the first n terms of the geometric sequence -/
def S (n : ℕ) (m : ℝ) : ℝ := m * 2^(n - 1) - 3

/-- Definition of the nth term of the geometric sequence -/
def a (n : ℕ) (m : ℝ) : ℝ :=
  if n = 1 then S 1 m
  else S n m - S (n - 1) m

/-- Theorem stating that m = 6 for the given geometric sequence -/
theorem geometric_sequence_m_value :
  ∃ (m : ℝ), ∀ (n : ℕ), n ≥ 1 → (a n m) / (a 1 m) = 2^(n - 1) ∧ m = 6 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_m_value_l3859_385915


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l3859_385944

theorem sequence_sum_problem (N : ℤ) : 
  (995 : ℤ) + 997 + 999 + 1001 + 1003 = 5005 - N → N = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l3859_385944


namespace NUMINAMATH_CALUDE_james_weight_plates_purchase_l3859_385987

/-- Represents the purchase of a weight vest and weight plates -/
structure WeightPurchase where
  vest_cost : ℝ
  plate_cost_per_pound : ℝ
  discounted_200lb_vest_cost : ℝ
  savings : ℝ

/-- Calculates the number of pounds of weight plates purchased -/
def weight_plates_purchased (purchase : WeightPurchase) : ℕ :=
  sorry

/-- Theorem stating that James purchased 291 pounds of weight plates -/
theorem james_weight_plates_purchase :
  let purchase : WeightPurchase := {
    vest_cost := 250,
    plate_cost_per_pound := 1.2,
    discounted_200lb_vest_cost := 700 - 100,
    savings := 110
  }
  weight_plates_purchased purchase = 291 := by
  sorry

end NUMINAMATH_CALUDE_james_weight_plates_purchase_l3859_385987
