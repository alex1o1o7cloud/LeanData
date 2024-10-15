import Mathlib

namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3043_304348

/-- The speed of a boat in still water, given its speed with and against a stream. -/
theorem boat_speed_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 36) 
  (h2 : speed_against_stream = 8) : 
  (speed_with_stream + speed_against_stream) / 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3043_304348


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l3043_304322

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The statement of the problem -/
theorem coin_flip_probability_difference : 
  prob_k_heads 4 3 - prob_k_heads 4 4 = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l3043_304322


namespace NUMINAMATH_CALUDE_segments_in_proportion_l3043_304310

/-- Four line segments are in proportion if the product of the outer segments
    equals the product of the inner segments -/
def are_in_proportion (a b c d : ℝ) : Prop :=
  a * d = b * c

/-- The set of line segments (4, 8, 5, 10) -/
def segment_set : Vector ℝ 4 := ⟨[4, 8, 5, 10], rfl⟩

/-- Theorem: The set of line segments (4, 8, 5, 10) is in proportion -/
theorem segments_in_proportion :
  are_in_proportion (segment_set.get 0) (segment_set.get 1) (segment_set.get 2) (segment_set.get 3) :=
by
  sorry

end NUMINAMATH_CALUDE_segments_in_proportion_l3043_304310


namespace NUMINAMATH_CALUDE_d_range_l3043_304306

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the function d
def d (P : ℝ × ℝ) : ℝ := distance P A + distance P B

-- Theorem statement
theorem d_range :
  ∀ P : ℝ × ℝ, C P.1 P.2 → 32 ≤ d P ∧ d P ≤ 72 :=
sorry

end NUMINAMATH_CALUDE_d_range_l3043_304306


namespace NUMINAMATH_CALUDE_outside_point_distance_l3043_304334

/-- A circle with center O and radius 5 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 5)

/-- A point P outside the circle -/
structure OutsidePoint (c : Circle) :=
  (P : ℝ × ℝ)
  (h_outside : dist P c.O > c.radius)

/-- The statement to prove -/
theorem outside_point_distance {c : Circle} (p : OutsidePoint c) :
  dist p.P c.O > 5 := by sorry

end NUMINAMATH_CALUDE_outside_point_distance_l3043_304334


namespace NUMINAMATH_CALUDE_expression_evaluation_l3043_304311

theorem expression_evaluation :
  ∀ x : ℝ, x = -2 → x * (x^2 - 4) = 0 →
  (x - 3) / (3 * x^2 - 6 * x) * (x + 2 - 5 / (x - 2)) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3043_304311


namespace NUMINAMATH_CALUDE_cryptarithm_no_solution_l3043_304386

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a mapping from characters to digits -/
def DigitAssignment := Char → Digit

/-- Checks if all characters in a string are mapped to unique digits -/
def all_unique (s : String) (assignment : DigitAssignment) : Prop :=
  ∀ c₁ c₂, c₁ ∈ s.data → c₂ ∈ s.data → c₁ ≠ c₂ → assignment c₁ ≠ assignment c₂

/-- Converts a string to a number using the given digit assignment -/
def to_number (s : String) (assignment : DigitAssignment) : ℕ :=
  s.foldl (fun acc c => 10 * acc + (assignment c).val) 0

/-- The main theorem stating that the cryptarithm has no solution -/
theorem cryptarithm_no_solution :
  ¬ ∃ (assignment : DigitAssignment),
    all_unique "DONAKLENVG" assignment ∧
    to_number "DON" assignment + to_number "OKA" assignment +
    to_number "LENA" assignment + to_number "VOLGA" assignment =
    to_number "ANGARA" assignment :=
by sorry


end NUMINAMATH_CALUDE_cryptarithm_no_solution_l3043_304386


namespace NUMINAMATH_CALUDE_Q_value_at_8_l3043_304363

-- Define the polynomial Q(x)
def Q (x : ℂ) (g h i j k l m : ℝ) : ℂ :=
  (3 * x^4 - 54 * x^3 + g * x^2 + h * x + i) *
  (4 * x^5 - 100 * x^4 + j * x^3 + k * x^2 + l * x + m)

-- Define the set of roots
def roots : Set ℂ := {2, 3, 4, 6, 7}

-- Theorem statement
theorem Q_value_at_8 (g h i j k l m : ℝ) :
  (∀ z : ℂ, Q z g h i j k l m = 0 → z ∈ roots) →
  Q 8 g h i j k l m = 14400 := by
  sorry


end NUMINAMATH_CALUDE_Q_value_at_8_l3043_304363


namespace NUMINAMATH_CALUDE_cos_105_degrees_l3043_304321

theorem cos_105_degrees : Real.cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l3043_304321


namespace NUMINAMATH_CALUDE_tshirt_price_is_8_l3043_304301

-- Define the prices and quantities
def sweater_price : ℝ := 18
def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.1
def sales_tax : ℝ := 0.05
def num_tshirts : ℕ := 6
def num_sweaters : ℕ := 4
def num_jackets : ℕ := 5
def total_cost : ℝ := 504

-- Define the function to calculate the total cost
def calculate_total_cost (tshirt_price : ℝ) : ℝ :=
  let jacket_price := jacket_original_price * (1 - jacket_discount)
  let subtotal := num_tshirts * tshirt_price + num_sweaters * sweater_price + num_jackets * jacket_price
  subtotal * (1 + sales_tax)

-- Theorem to prove
theorem tshirt_price_is_8 :
  ∃ (tshirt_price : ℝ), calculate_total_cost tshirt_price = total_cost ∧ tshirt_price = 8 :=
sorry

end NUMINAMATH_CALUDE_tshirt_price_is_8_l3043_304301


namespace NUMINAMATH_CALUDE_real_part_of_one_plus_i_squared_l3043_304356

theorem real_part_of_one_plus_i_squared (i : ℂ) : 
  Complex.re ((1 + i)^2) = 0 := by sorry

end NUMINAMATH_CALUDE_real_part_of_one_plus_i_squared_l3043_304356


namespace NUMINAMATH_CALUDE_train_journey_duration_l3043_304324

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference in minutes between two times -/
def timeDifferenceInMinutes (t1 t2 : Time) : Nat :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Represents the state of clock hands -/
inductive ClockHandState
  | Symmetrical
  | NotSymmetrical

theorem train_journey_duration (stationArrival : Time)
                               (trainDeparture : Time)
                               (destinationArrival : Time)
                               (stationDeparture : Time)
                               (boardingState : ClockHandState)
                               (alightingState : ClockHandState) :
  stationArrival = ⟨8, 0⟩ →
  trainDeparture = ⟨8, 35⟩ →
  destinationArrival = ⟨14, 15⟩ →
  stationDeparture = ⟨15, 0⟩ →
  boardingState = ClockHandState.Symmetrical →
  alightingState = ClockHandState.Symmetrical →
  timeDifferenceInMinutes trainDeparture stationDeparture = 385 :=
by sorry

end NUMINAMATH_CALUDE_train_journey_duration_l3043_304324


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3043_304331

theorem rectangle_area_change (l w : ℝ) (h : l * w = 1100) :
  (1.1 * l) * (0.9 * w) = 1089 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3043_304331


namespace NUMINAMATH_CALUDE_furniture_dealer_tables_l3043_304391

/-- The number of four-legged tables -/
def F : ℕ := 16

/-- The number of three-legged tables -/
def T : ℕ := (124 - 4 * F) / 3

/-- The total number of tables -/
def total_tables : ℕ := F + T

/-- Theorem stating that the total number of tables is 36 -/
theorem furniture_dealer_tables : total_tables = 36 := by
  sorry

end NUMINAMATH_CALUDE_furniture_dealer_tables_l3043_304391


namespace NUMINAMATH_CALUDE_minimum_tip_percentage_l3043_304325

theorem minimum_tip_percentage
  (meal_cost : ℝ)
  (total_paid : ℝ)
  (h_meal_cost : meal_cost = 35.50)
  (h_total_paid : total_paid = 37.275)
  (h_tip_less_than_8 : (total_paid - meal_cost) / meal_cost < 0.08) :
  (total_paid - meal_cost) / meal_cost = 0.05 :=
by sorry

end NUMINAMATH_CALUDE_minimum_tip_percentage_l3043_304325


namespace NUMINAMATH_CALUDE_five_cuts_sixteen_pieces_l3043_304392

/-- The number of pieces obtained by cutting a cake n times -/
def cakePieces (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem: The number of pieces obtained by cutting a cake 5 times is 16 -/
theorem five_cuts_sixteen_pieces : cakePieces 5 = 16 := by
  sorry

#eval cakePieces 5  -- This will evaluate to 16

end NUMINAMATH_CALUDE_five_cuts_sixteen_pieces_l3043_304392


namespace NUMINAMATH_CALUDE_gas_volume_at_10_degrees_l3043_304336

-- Define the relationship between temperature change and volume change
def volume_change (temp_change : ℤ) : ℤ := (3 * temp_change) / 5

-- Define the initial conditions
def initial_temp : ℤ := 25
def initial_volume : ℤ := 40
def final_temp : ℤ := 10

-- Define the theorem
theorem gas_volume_at_10_degrees : 
  initial_volume + volume_change (final_temp - initial_temp) = 31 := by
  sorry

end NUMINAMATH_CALUDE_gas_volume_at_10_degrees_l3043_304336


namespace NUMINAMATH_CALUDE_sets_equal_implies_a_value_l3043_304395

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | -1 ≤ x ∧ x ≤ a}
def B (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = x + 1}
def C (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = x^2}

-- State the theorem
theorem sets_equal_implies_a_value (a : ℝ) (h1 : a > -1) (h2 : B a = C a) :
  a = 0 ∨ a = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sets_equal_implies_a_value_l3043_304395


namespace NUMINAMATH_CALUDE_flip_colors_iff_even_l3043_304317

/-- Represents the color of a square on the board -/
inductive Color
| White
| Black
| Orange

/-- Represents a 3n × 3n board -/
def Board (n : ℕ) := Fin (3*n) → Fin (3*n) → Color

/-- Initial coloring of the board -/
def initialBoard (n : ℕ) : Board n :=
  λ i j => if (i.val + j.val) % 3 = 2 then Color.Black else Color.White

/-- A move on the board -/
def move (b : Board n) (i j : Fin (3*n)) : Board n :=
  λ x y => if x.val ∈ [i.val, i.val+1] ∧ y.val ∈ [j.val, j.val+1]
           then match b x y with
                | Color.White => Color.Orange
                | Color.Orange => Color.Black
                | Color.Black => Color.White
           else b x y

/-- The goal state of the board -/
def goalBoard (n : ℕ) : Board n :=
  λ i j => if (i.val + j.val) % 3 = 2 then Color.White else Color.Black

/-- A sequence of moves -/
def MoveSequence (n : ℕ) := List (Fin (3*n) × Fin (3*n))

/-- Apply a sequence of moves to a board -/
def applyMoves (b : Board n) (moves : MoveSequence n) : Board n :=
  moves.foldl (λ board (i, j) => move board i j) b

theorem flip_colors_iff_even (n : ℕ) (h : n > 0) :
  (∃ (moves : MoveSequence n), applyMoves (initialBoard n) moves = goalBoard n) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_flip_colors_iff_even_l3043_304317


namespace NUMINAMATH_CALUDE_scooter_cost_recovery_l3043_304357

/-- The minimum number of deliveries required to recover the initial cost of a scooter -/
def min_deliveries (initial_cost earnings fuel_cost parking_fee : ℕ) : ℕ :=
  (initial_cost + (earnings - fuel_cost - parking_fee) - 1) / (earnings - fuel_cost - parking_fee)

/-- Theorem stating the minimum number of deliveries required to recover the scooter cost -/
theorem scooter_cost_recovery :
  min_deliveries 3000 12 4 1 = 429 := by
  sorry

end NUMINAMATH_CALUDE_scooter_cost_recovery_l3043_304357


namespace NUMINAMATH_CALUDE_mode_median_determinable_l3043_304332

/-- Represents the age distribution of students in the model aviation interest group --/
structure AgeDistribution where
  total : Nat
  age13 : Nat
  age14 : Nat
  age15 : Nat
  age16 : Nat

/-- Conditions of the problem --/
def aviation_group : AgeDistribution where
  total := 50
  age13 := 5
  age14 := 23
  age15 := 0  -- Unknown, represented as 0
  age16 := 0  -- Unknown, represented as 0

/-- Definition of mode --/
def mode (ad : AgeDistribution) : Nat :=
  max (max ad.age13 ad.age14) (max ad.age15 ad.age16)

/-- Definition of median for even number of students --/
def median (ad : AgeDistribution) : Nat :=
  if ad.age13 + ad.age14 ≥ ad.total / 2 then 14 else 15

/-- Main theorem --/
theorem mode_median_determinable (ad : AgeDistribution) 
  (h1 : ad.total = 50)
  (h2 : ad.age13 = 5)
  (h3 : ad.age14 = 23)
  (h4 : ad.age15 + ad.age16 = ad.total - ad.age13 - ad.age14) :
  (∃ (m : Nat), mode ad = m) ∧ 
  (∃ (n : Nat), median ad = n) ∧
  (¬ ∃ (mean : ℚ), true) ∧  -- Mean cannot be determined
  (¬ ∃ (variance : ℚ), true) :=  -- Variance cannot be determined
sorry


end NUMINAMATH_CALUDE_mode_median_determinable_l3043_304332


namespace NUMINAMATH_CALUDE_actual_car_mass_is_1331_l3043_304318

/-- The mass of a scaled model car -/
def model_mass : ℝ := 1

/-- The scale factor between the model and the actual car -/
def scale_factor : ℝ := 11

/-- Calculates the mass of the actual car given the model mass and scale factor -/
def actual_car_mass (model_mass : ℝ) (scale_factor : ℝ) : ℝ :=
  model_mass * (scale_factor ^ 3)

/-- Theorem stating that the mass of the actual car is 1331 kg -/
theorem actual_car_mass_is_1331 :
  actual_car_mass model_mass scale_factor = 1331 := by
  sorry

end NUMINAMATH_CALUDE_actual_car_mass_is_1331_l3043_304318


namespace NUMINAMATH_CALUDE_lines_coincide_by_rotation_l3043_304377

/-- Given two lines l₁ and l₂ in the plane, prove that they can coincide by rotation -/
theorem lines_coincide_by_rotation (α c : ℝ) :
  ∃ (x₀ y₀ θ : ℝ), 
    (y₀ = x₀ * Real.sin α) ∧  -- Point (x₀, y₀) is on l₁
    (∀ x y : ℝ, 
      y = x * Real.sin α →  -- Original line l₁
      ∃ x' y' : ℝ, 
        x' = (x - x₀) * Real.cos θ - (y - y₀) * Real.sin θ + x₀ ∧
        y' = (x - x₀) * Real.sin θ + (y - y₀) * Real.cos θ + y₀ ∧
        y' = 2 * x' + c)  -- Rotated line coincides with l₂
  := by sorry

end NUMINAMATH_CALUDE_lines_coincide_by_rotation_l3043_304377


namespace NUMINAMATH_CALUDE_ladder_problem_l3043_304384

theorem ladder_problem (ladder_length height base : ℝ) :
  ladder_length = 15 ∧ height = 12 ∧ ladder_length ^ 2 = height ^ 2 + base ^ 2 →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l3043_304384


namespace NUMINAMATH_CALUDE_jerrys_age_l3043_304333

/-- Given that Mickey's age is 18 and Mickey's age is 4 years less than 400% of Jerry's age,
    prove that Jerry's age is 5.5 years. -/
theorem jerrys_age (mickey_age jerry_age : ℝ) : 
  mickey_age = 18 ∧ 
  mickey_age = 4 * jerry_age - 4 → 
  jerry_age = 5.5 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l3043_304333


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3043_304362

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x ↦ c * x^3 - 4 * x^2 + d * x - 7
  (g 2 = -7) ∧ (g (-1) = -20) → c = -1/3 ∧ d = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3043_304362


namespace NUMINAMATH_CALUDE_line_plane_intersection_l3043_304335

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Set Line)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define intersection relation for lines
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem line_plane_intersection 
  (m n : Line) (α β : Plane) :
  (intersect α β = {m} ∧ subset n α) →
  (parallel m n ∨ intersects m n) :=
sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l3043_304335


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3043_304342

-- Define the quadratic function f(x)
def f (x : ℝ) := 2 * x^2 - 10 * x

-- State the theorem
theorem quadratic_function_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 4, f x ≤ 12) ∧  -- maximum value is 12 on [-1,4]
  (∀ x : ℝ, f x < 0 ↔ x ∈ Set.Ioo 0 5) ∧  -- solution set of f(x) < 0 is (0,5)
  (∀ x m : ℝ, m < -5 ∨ m > 1 → f (2 - 2 * Real.cos x) < f (1 - Real.cos x - m)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3043_304342


namespace NUMINAMATH_CALUDE_product_of_six_consecutive_numbers_l3043_304359

theorem product_of_six_consecutive_numbers (n : ℕ) (h : n = 3) :
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_of_six_consecutive_numbers_l3043_304359


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l3043_304399

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l3043_304399


namespace NUMINAMATH_CALUDE_range_of_f_l3043_304376

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ∈ domain, f x = y} = {y : ℝ | 2 ≤ y ∧ y ≤ 6} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3043_304376


namespace NUMINAMATH_CALUDE_f_extrema_l3043_304375

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

theorem f_extrema :
  let a : ℝ := -1
  let b : ℝ := (Real.exp 2 - 3) / 2
  (∀ x ∈ Set.Icc a b, f (-1/2) ≤ f x) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ f ((Real.exp 2 - 3) / 2)) ∧
  f (-1/2) = Real.log 2 + 1/4 ∧
  f ((Real.exp 2 - 3) / 2) = 2 + (Real.exp 2 - 3)^2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l3043_304375


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3043_304349

theorem inequality_system_solutions :
  let S := {x : ℤ | (3 * x + 1 < x - 3) ∧ ((1 + x) / 2 ≤ (1 + 2 * x) / 3 + 1)}
  S = {-5, -4, -3} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3043_304349


namespace NUMINAMATH_CALUDE_current_speed_l3043_304320

/-- Given a boat's upstream and downstream speeds, calculate the speed of the current --/
theorem current_speed (v_upstream v_downstream : ℝ) (h1 : v_upstream = 2) (h2 : v_downstream = 5) :
  (v_downstream - v_upstream) / 2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l3043_304320


namespace NUMINAMATH_CALUDE_royal_family_children_l3043_304398

/-- Represents the number of years that have passed -/
def n : ℕ := sorry

/-- Represents the number of daughters -/
def d : ℕ := sorry

/-- The initial age of the king and queen -/
def initial_parent_age : ℕ := 35

/-- The initial total age of the children -/
def initial_children_age : ℕ := 35

/-- The number of sons -/
def num_sons : ℕ := 3

/-- The maximum allowed number of children -/
def max_children : ℕ := 20

theorem royal_family_children :
  (initial_parent_age * 2 + 2 * n = initial_children_age + (d + num_sons) * n) ∧
  (d + num_sons ≤ max_children) →
  (d + num_sons = 7) ∨ (d + num_sons = 9) := by
  sorry

end NUMINAMATH_CALUDE_royal_family_children_l3043_304398


namespace NUMINAMATH_CALUDE_min_attendees_with_both_l3043_304387

theorem min_attendees_with_both (n : ℕ) (h1 : n > 0) : ∃ x : ℕ,
  x ≥ 1 ∧
  x ≤ n ∧
  x ≤ n / 3 ∧
  x ≤ n / 2 ∧
  ∀ y : ℕ, (y < x → ¬(y ≤ n / 3 ∧ y ≤ n / 2)) :=
by
  sorry

#check min_attendees_with_both

end NUMINAMATH_CALUDE_min_attendees_with_both_l3043_304387


namespace NUMINAMATH_CALUDE_binomial_505_505_equals_1_l3043_304329

theorem binomial_505_505_equals_1 : Nat.choose 505 505 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_505_505_equals_1_l3043_304329


namespace NUMINAMATH_CALUDE_military_unit_march_speeds_l3043_304343

/-- Proves that given the conditions of the military unit's march, the average speeds on the first and second days are 12 km/h and 10 km/h respectively. -/
theorem military_unit_march_speeds :
  ∀ (speed_day1 speed_day2 : ℝ),
    4 * speed_day1 + 5 * speed_day2 = 98 →
    4 * speed_day1 = 5 * speed_day2 - 2 →
    speed_day1 = 12 ∧ speed_day2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_military_unit_march_speeds_l3043_304343


namespace NUMINAMATH_CALUDE_tree_distance_l3043_304316

/-- Given 10 equally spaced trees along a road, with 100 feet between the 1st and 5th tree,
    the distance between the 1st and 10th tree is 225 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 10) (h2 : d = 100) :
  let space := d / 4
  (n - 1) * space = 225 :=
by sorry

end NUMINAMATH_CALUDE_tree_distance_l3043_304316


namespace NUMINAMATH_CALUDE_min_value_f_range_of_m_l3043_304371

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x + 2

def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x

theorem min_value_f (t : ℝ) (ht : t > 0) :
  (if t ≥ 1/Real.exp 1 then
    IsMinOn f (Set.Icc t (t + 2)) (f t)
   else
    IsMinOn f (Set.Icc t (t + 2)) (f (1/Real.exp 1))) ∧
  (if t ≥ 1/Real.exp 1 then
    ∀ x ∈ Set.Icc t (t + 2), f x ≥ t * Real.log t + 2
   else
    ∀ x ∈ Set.Icc t (t + 2), f x ≥ -1/Real.exp 1 + 2) :=
sorry

theorem range_of_m :
  {m : ℝ | ∃ x₀ ∈ Set.Icc (1/Real.exp 1) (Real.exp 1),
    m * (Real.log x₀ + 1) + g m x₀ ≥ 2*x₀ + m} = Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_min_value_f_range_of_m_l3043_304371


namespace NUMINAMATH_CALUDE_smallest_number_l3043_304361

theorem smallest_number (a b c d : ℝ) : 
  a = -2 → b = 4 → c = -5 → d = 1 → 
  (c < -3 ∧ a > -3 ∧ b > -3 ∧ d > -3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3043_304361


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l3043_304378

-- Define the function
def f (x : ℝ) : ℝ := x^(2/3)

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l3043_304378


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l3043_304385

theorem quadratic_complex_roots
  (a b c : ℝ) (x : ℂ)
  (h_a : a ≠ 0)
  (h_root : a * (1 + Complex.I)^2 + b * (1 + Complex.I) + c = 0) :
  a * (1 - Complex.I)^2 + b * (1 - Complex.I) + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l3043_304385


namespace NUMINAMATH_CALUDE_bullet_speed_difference_l3043_304328

/-- The speed of the horse in feet per second -/
def horse_speed : ℝ := 20

/-- The speed of the bullet in feet per second -/
def bullet_speed : ℝ := 400

/-- The difference in bullet speed when fired in the same direction as the horse's movement
    versus the opposite direction -/
def speed_difference : ℝ := (bullet_speed + horse_speed) - (bullet_speed - horse_speed)

theorem bullet_speed_difference :
  speed_difference = 40 := by sorry

end NUMINAMATH_CALUDE_bullet_speed_difference_l3043_304328


namespace NUMINAMATH_CALUDE_pythagorean_triple_9_12_15_l3043_304323

theorem pythagorean_triple_9_12_15 : 9^2 + 12^2 = 15^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_9_12_15_l3043_304323


namespace NUMINAMATH_CALUDE_three_digit_numbers_from_4_and_5_l3043_304379

def is_valid_digit (d : ℕ) : Prop := d = 4 ∨ d = 5

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_formed_from_4_and_5 (n : ℕ) : Prop :=
  is_three_digit_number n ∧
  is_valid_digit (n / 100) ∧
  is_valid_digit ((n / 10) % 10) ∧
  is_valid_digit (n % 10)

def valid_numbers : Finset ℕ :=
  {444, 445, 454, 455, 544, 545, 554, 555}

theorem three_digit_numbers_from_4_and_5 :
  ∀ n : ℕ, is_formed_from_4_and_5 n ↔ n ∈ valid_numbers :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_from_4_and_5_l3043_304379


namespace NUMINAMATH_CALUDE_graph_passes_through_quadrants_l3043_304389

-- Define the function
def f (x : ℝ) : ℝ := -3 * x + 1

-- Theorem statement
theorem graph_passes_through_quadrants :
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- First quadrant
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧  -- Second quadrant
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) :=  -- Fourth quadrant
by sorry

end NUMINAMATH_CALUDE_graph_passes_through_quadrants_l3043_304389


namespace NUMINAMATH_CALUDE_rectangular_hall_area_l3043_304305

theorem rectangular_hall_area (length width : ℝ) : 
  width = (1/2) * length →
  length - width = 12 →
  length * width = 288 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_area_l3043_304305


namespace NUMINAMATH_CALUDE_proposition_and_equivalents_l3043_304326

def IsDecreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≥ a (n + 1)

theorem proposition_and_equivalents (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n ↔ IsDecreasing a) ∧
  (IsDecreasing a ↔ ∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) ∧
  (∀ n : ℕ+, (a n + a (n + 1)) / 2 ≥ a n ↔ ¬IsDecreasing a) ∧
  (¬IsDecreasing a ↔ ∀ n : ℕ+, (a n + a (n + 1)) / 2 ≥ a n) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_equivalents_l3043_304326


namespace NUMINAMATH_CALUDE_car_trip_local_road_distance_l3043_304355

theorem car_trip_local_road_distance 
  (local_speed highway_speed avg_speed : ℝ)
  (highway_distance : ℝ)
  (local_speed_pos : local_speed > 0)
  (highway_speed_pos : highway_speed > 0)
  (avg_speed_pos : avg_speed > 0)
  (highway_distance_pos : highway_distance > 0)
  (h_local_speed : local_speed = 20)
  (h_highway_speed : highway_speed = 60)
  (h_highway_distance : highway_distance = 120)
  (h_avg_speed : avg_speed = 36) :
  ∃ (local_distance : ℝ),
    local_distance > 0 ∧
    (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = avg_speed ∧
    local_distance = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_local_road_distance_l3043_304355


namespace NUMINAMATH_CALUDE_roberto_outfits_l3043_304397

/-- The number of different outfits Roberto can put together -/
def number_of_outfits (trousers shirts jackets belts : ℕ) 
  (restricted_jacket_trousers : ℕ) : ℕ :=
  let unrestricted_jackets := jackets - 1
  let unrestricted_combinations := trousers * shirts * unrestricted_jackets * belts
  let restricted_combinations := restricted_jacket_trousers * shirts * belts
  let overlapping_combinations := (trousers - restricted_jacket_trousers) * shirts * belts
  unrestricted_combinations + restricted_combinations - overlapping_combinations

/-- Theorem stating the number of outfits Roberto can put together -/
theorem roberto_outfits : 
  number_of_outfits 5 7 4 2 3 = 168 := by
  sorry

#eval number_of_outfits 5 7 4 2 3

end NUMINAMATH_CALUDE_roberto_outfits_l3043_304397


namespace NUMINAMATH_CALUDE_parabola_equation_l3043_304327

/-- The equation of a parabola with focus at the center of x^2 + y^2 = 4x and vertex at origin -/
theorem parabola_equation (x y : ℝ) :
  (∃ (c : ℝ × ℝ), c.1^2 + c.2^2 = 4*c.1 ∧ 
   (x - c.1)^2 + (y - c.2)^2 = (x - 0)^2 + (y - 0)^2) →
  y^2 = 8*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3043_304327


namespace NUMINAMATH_CALUDE_smallest_h_divisible_by_primes_l3043_304314

theorem smallest_h_divisible_by_primes : ∃ (h : ℕ), h > 0 ∧ 
  (∀ (h' : ℕ), h' < h → ¬∃ (k : ℤ), (13 ∣ (h' + k)) ∧ (17 ∣ (h' + k)) ∧ (29 ∣ (h' + k))) ∧
  ∃ (k : ℤ), (13 ∣ (h + k)) ∧ (17 ∣ (h + k)) ∧ (29 ∣ (h + k)) :=
by sorry

#check smallest_h_divisible_by_primes

end NUMINAMATH_CALUDE_smallest_h_divisible_by_primes_l3043_304314


namespace NUMINAMATH_CALUDE_distance_CD_l3043_304308

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 16 * (x - 3)^2 + 4 * (y + 2)^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -2)

-- Define the length of the semi-major axis
def a : ℝ := 4

-- Define the length of the semi-minor axis
def b : ℝ := 2

-- Define an endpoint of the major axis
def C : ℝ × ℝ := (center.1 + a, center.2)

-- Define an endpoint of the minor axis
def D : ℝ × ℝ := (center.1, center.2 + b)

-- Theorem statement
theorem distance_CD : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_CD_l3043_304308


namespace NUMINAMATH_CALUDE_characterize_function_l3043_304374

theorem characterize_function (f : ℤ → ℤ) :
  (∀ x y z : ℤ, x + y + z = 0 → f x + f y + f z = x * y * z) →
  ∃ c : ℤ, ∀ x : ℤ, f x = (x^3 - x) / 3 + c * x :=
sorry

end NUMINAMATH_CALUDE_characterize_function_l3043_304374


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3043_304341

theorem sum_of_reciprocal_equations (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : 1/x - 1/y = -1) : 
  x + y = 5/6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l3043_304341


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3043_304347

theorem rectangle_perimeter (l w : ℝ) (h1 : l > 0) (h2 : w > 0) : 
  l * w = 360 ∧ (l + 10) * (w - 6) = 360 → 2 * (l + w) = 76 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3043_304347


namespace NUMINAMATH_CALUDE_parallel_vectors_m_zero_l3043_304312

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_zero :
  let a : ℝ × ℝ := (-2, 3)
  let b : ℝ × ℝ := (1, m - 3/2)
  parallel a b → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_zero_l3043_304312


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3043_304309

/-- The minimum value of a quadratic function f(x) = ax^2 + (b + 5)x + c where a > 0 -/
theorem quadratic_minimum_value (a b c : ℝ) (ha : a > 0) :
  let f := fun x => a * x^2 + (b + 5) * x + c
  ∃ m, ∀ x, f x ≥ m ∧ ∃ x₀, f x₀ = m :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3043_304309


namespace NUMINAMATH_CALUDE_cone_with_hole_volume_l3043_304302

/-- The volume of a cone with a cylindrical hole -/
theorem cone_with_hole_volume
  (cone_diameter : ℝ)
  (cone_height : ℝ)
  (hole_diameter : ℝ)
  (h_cone_diameter : cone_diameter = 12)
  (h_cone_height : cone_height = 12)
  (h_hole_diameter : hole_diameter = 4) :
  (1/3 * π * (cone_diameter/2)^2 * cone_height) - (π * (hole_diameter/2)^2 * cone_height) = 96 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_with_hole_volume_l3043_304302


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3043_304368

-- Define the hyperbolas
def C₁ (x y : ℝ) : Prop := x^2/4 - y^2/3 = 1
def C₂ (x y : ℝ) : Prop := x^2/4 - y^2/3 = -1

-- Define focal length
def focal_length (C : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Define foci
def foci (C : (ℝ → ℝ → Prop)) : Set (ℝ × ℝ) := sorry

-- Define asymptotic lines
def asymptotic_lines (C : (ℝ → ℝ → Prop)) : Set (ℝ → ℝ → Prop) := sorry

-- Define eccentricity
def eccentricity (C : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem hyperbola_properties :
  (focal_length C₁ = focal_length C₂) ∧
  (∃ (r : ℝ), ∀ (p : ℝ × ℝ), p ∈ foci C₁ ∪ foci C₂ → p.1^2 + p.2^2 = r^2) ∧
  (asymptotic_lines C₁ = asymptotic_lines C₂) ∧
  (eccentricity C₁ ≠ eccentricity C₂) := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3043_304368


namespace NUMINAMATH_CALUDE_factorization_sum_l3043_304346

theorem factorization_sum (A B C D E F G H J K : ℤ) (x y : ℚ) : 
  (125 * x^8 - 2401 * y^8 = (A * x + B * y) * (C * x^4 + D * x * y + E * y^4) * 
                            (F * x + G * y) * (H * x^4 + J * x * y + K * y^4)) →
  A + B + C + D + E + F + G + H + J + K = 102 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l3043_304346


namespace NUMINAMATH_CALUDE_base_ten_to_four_156_base_four_to_ten_2130_l3043_304313

/-- Converts a natural number from base 10 to base 4 --/
def toBaseFour (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Converts a list of digits in base 4 to a natural number in base 10 --/
def fromBaseFour (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ (digits.length - 1 - i))) 0

theorem base_ten_to_four_156 : toBaseFour 156 = [2, 1, 3, 0] := by sorry

theorem base_four_to_ten_2130 : fromBaseFour [2, 1, 3, 0] = 156 := by sorry

end NUMINAMATH_CALUDE_base_ten_to_four_156_base_four_to_ten_2130_l3043_304313


namespace NUMINAMATH_CALUDE_union_M_N_equals_interval_l3043_304340

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | (x + 1) * (x - 3) < 0}

-- Define the interval (-1, +∞)
def openIntervalFromNegativeOneToInfinity : Set ℝ := {x : ℝ | x > -1}

-- State the theorem
theorem union_M_N_equals_interval : M ∪ N = openIntervalFromNegativeOneToInfinity := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_equals_interval_l3043_304340


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l3043_304303

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  black : ℕ

/-- Calculates the probability of drawing two red balls -/
def prob_two_red (bag : ColoredBalls) : ℚ :=
  (bag.red.choose 2 : ℚ) / (bag.total.choose 2)

/-- Calculates the probability of drawing two balls of different colors -/
def prob_different_colors (bag : ColoredBalls) : ℚ :=
  (bag.red * bag.black : ℚ) / (bag.total.choose 2)

/-- The main theorem about probabilities in the ball drawing scenario -/
theorem ball_drawing_probabilities (bag : ColoredBalls) 
    (h_total : bag.total = 6)
    (h_red : bag.red = 4)
    (h_black : bag.black = 2) :
    prob_two_red bag = 2/5 ∧ prob_different_colors bag = 8/15 := by
  sorry


end NUMINAMATH_CALUDE_ball_drawing_probabilities_l3043_304303


namespace NUMINAMATH_CALUDE_circle_max_area_center_l3043_304364

/-- Given a circle with equation x^2 + y^2 + kx + 2y + k^2 = 0,
    prove that its center is (0, -1) when the area is maximum. -/
theorem circle_max_area_center (k : ℝ) :
  let circle_eq := λ (x y : ℝ) => x^2 + y^2 + k*x + 2*y + k^2 = 0
  let center := (-(k/2), -1)
  let radius_squared := 1 - (3/4) * k^2
  (∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius_squared) →
  (radius_squared ≤ 1) →
  (radius_squared = 1 ↔ k = 0) →
  (k = 0 → center = (0, -1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_max_area_center_l3043_304364


namespace NUMINAMATH_CALUDE_apples_handed_out_to_students_l3043_304366

theorem apples_handed_out_to_students 
  (initial_apples : ℕ) 
  (pies_made : ℕ) 
  (apples_per_pie : ℕ) 
  (h1 : initial_apples = 62)
  (h2 : pies_made = 6)
  (h3 : apples_per_pie = 9) :
  initial_apples - (pies_made * apples_per_pie) = 8 := by
sorry

end NUMINAMATH_CALUDE_apples_handed_out_to_students_l3043_304366


namespace NUMINAMATH_CALUDE_sum_of_factors_l3043_304300

theorem sum_of_factors (a b c : ℤ) : 
  (∀ x, x^2 + 20*x + 96 = (x + a) * (x + b)) →
  (∀ x, x^2 + 18*x + 81 = (x - b) * (x + c)) →
  a + b + c = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l3043_304300


namespace NUMINAMATH_CALUDE_reduced_rate_end_time_l3043_304354

/-- Represents the fraction of a week with reduced rates -/
def reduced_rate_fraction : ℚ := 0.6428571428571429

/-- Represents the number of hours in a week -/
def hours_in_week : ℕ := 7 * 24

/-- Represents the number of hours with reduced rates on weekends -/
def weekend_reduced_hours : ℕ := 2 * 24

/-- Represents the hour when reduced rates start on weekdays (24-hour format) -/
def weekday_start_hour : ℕ := 20

/-- Represents the hour when reduced rates end on weekdays (24-hour format) -/
def weekday_end_hour : ℕ := 8

theorem reduced_rate_end_time :
  (reduced_rate_fraction * hours_in_week).floor - weekend_reduced_hours = 
  5 * (24 - weekday_start_hour + weekday_end_hour) :=
sorry

end NUMINAMATH_CALUDE_reduced_rate_end_time_l3043_304354


namespace NUMINAMATH_CALUDE_closest_point_l3043_304367

def v (s : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 3 + 5*s
  | 1 => -2 + 3*s
  | 2 => -4 - 2*s

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 5
  | 2 => 6

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 3
  | 2 => -2

theorem closest_point (s : ℝ) : 
  (∀ t : ℝ, ‖v t - a‖ ≥ ‖v s - a‖) ↔ s = 11/38 :=
sorry

end NUMINAMATH_CALUDE_closest_point_l3043_304367


namespace NUMINAMATH_CALUDE_exam_average_l3043_304390

theorem exam_average (total_boys : ℕ) (passed_boys : ℕ) (passed_avg : ℚ) (failed_avg : ℚ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 115)
  (h3 : passed_avg = 39)
  (h4 : failed_avg = 15) :
  (passed_boys * passed_avg + (total_boys - passed_boys) * failed_avg) / total_boys = 38 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l3043_304390


namespace NUMINAMATH_CALUDE_square_sum_geq_root3_product_l3043_304370

theorem square_sum_geq_root3_product (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_product_leq_sum : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_root3_product_l3043_304370


namespace NUMINAMATH_CALUDE_prism_with_nine_faces_has_fourteen_vertices_l3043_304380

/-- A prism is a polyhedron with two congruent polygon bases and rectangular lateral faces. -/
structure Prism where
  num_faces : ℕ
  num_base_sides : ℕ
  num_vertices : ℕ

/-- The number of faces in a prism is related to the number of sides in its base. -/
axiom prism_faces (p : Prism) : p.num_faces = p.num_base_sides + 2

/-- The number of vertices in a prism is twice the number of sides in its base. -/
axiom prism_vertices (p : Prism) : p.num_vertices = 2 * p.num_base_sides

/-- Theorem: A prism with 9 faces has 14 vertices. -/
theorem prism_with_nine_faces_has_fourteen_vertices :
  ∃ (p : Prism), p.num_faces = 9 ∧ p.num_vertices = 14 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_nine_faces_has_fourteen_vertices_l3043_304380


namespace NUMINAMATH_CALUDE_special_triangle_properties_l3043_304381

open Real

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  angle_sum : A + B + C = π

-- Define the specific conditions of the triangle
def SpecialTriangle (t : AcuteTriangle) : Prop :=
  t.B = 2 * t.A ∧ sin t.A ≠ 0 ∧ cos t.A ≠ 0

-- State the theorems to be proved
theorem special_triangle_properties (t : AcuteTriangle) (h : SpecialTriangle t) :
  ∃ (AC : ℝ), 
    AC / cos t.A = 2 ∧ 
    sqrt 2 < AC ∧ 
    AC < sqrt 3 := by sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l3043_304381


namespace NUMINAMATH_CALUDE_function_inequality_range_l3043_304393

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem function_inequality_range (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → 
    |f a x₁ - f a x₂| ≤ a - 2) ↔ 
  a ∈ Set.Ici (Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_range_l3043_304393


namespace NUMINAMATH_CALUDE_total_students_is_fifteen_l3043_304383

/-- The number of students originally in Class 1 -/
def n1 : ℕ := 8

/-- The number of students originally in Class 2 -/
def n2 : ℕ := 5

/-- Lei Lei's height in cm -/
def lei_lei_height : ℕ := 158

/-- Rong Rong's height in cm -/
def rong_rong_height : ℕ := 140

/-- The change in average height of Class 1 after the swap (in cm) -/
def class1_avg_change : ℚ := 2

/-- The change in average height of Class 2 after the swap (in cm) -/
def class2_avg_change : ℚ := 3

/-- The total number of students in both classes -/
def total_students : ℕ := n1 + n2 + 2

theorem total_students_is_fifteen :
  (lei_lei_height - rong_rong_height : ℚ) / (n1 + 1) = class1_avg_change ∧
  (lei_lei_height - rong_rong_height : ℚ) / (n2 + 1) = class2_avg_change →
  total_students = 15 := by sorry

end NUMINAMATH_CALUDE_total_students_is_fifteen_l3043_304383


namespace NUMINAMATH_CALUDE_vacation_fund_adjustment_l3043_304353

/-- Calculates the required weekly hours to meet a financial goal after losing one week of work --/
def required_hours (original_weeks : ℕ) (original_hours_per_week : ℕ) (total_earnings : ℕ) : ℚ :=
  let remaining_weeks := original_weeks - 1
  let hourly_rate := (total_earnings : ℚ) / (original_weeks * original_hours_per_week)
  let weekly_earnings_needed := (total_earnings : ℚ) / remaining_weeks
  weekly_earnings_needed / hourly_rate

theorem vacation_fund_adjustment (original_weeks : ℕ) (original_hours_per_week : ℕ) (total_earnings : ℕ) 
    (h1 : original_weeks = 10)
    (h2 : original_hours_per_week = 25)
    (h3 : total_earnings = 2500) :
  ∃ (n : ℕ), n ≤ required_hours original_weeks original_hours_per_week total_earnings ∧ 
             required_hours original_weeks original_hours_per_week total_earnings < n + 1 ∧
             n = 28 :=
  sorry

end NUMINAMATH_CALUDE_vacation_fund_adjustment_l3043_304353


namespace NUMINAMATH_CALUDE_alloy_mixture_percentage_l3043_304388

/-- Proves that mixing 66 ounces of 10% alloy with 55 ounces of 21% alloy
    results in 121 ounces of an alloy with 15% copper content. -/
theorem alloy_mixture_percentage :
  let alloy_10_amount : ℝ := 66
  let alloy_10_percentage : ℝ := 10
  let alloy_21_amount : ℝ := 55
  let alloy_21_percentage : ℝ := 21
  let total_amount : ℝ := alloy_10_amount + alloy_21_amount
  let total_copper : ℝ := (alloy_10_amount * alloy_10_percentage / 100) +
                          (alloy_21_amount * alloy_21_percentage / 100)
  let final_percentage : ℝ := total_copper / total_amount * 100
  total_amount = 121 ∧ final_percentage = 15 := by sorry

end NUMINAMATH_CALUDE_alloy_mixture_percentage_l3043_304388


namespace NUMINAMATH_CALUDE_fish_ratio_problem_l3043_304372

/-- The ratio of tagged fish to total fish in a second catch -/
def fish_ratio (tagged_initial : ℕ) (second_catch : ℕ) (tagged_in_catch : ℕ) (total_fish : ℕ) : ℚ :=
  tagged_in_catch / second_catch

/-- Theorem stating the ratio of tagged fish to total fish in the second catch -/
theorem fish_ratio_problem :
  let tagged_initial : ℕ := 30
  let second_catch : ℕ := 50
  let tagged_in_catch : ℕ := 2
  let total_fish : ℕ := 750
  fish_ratio tagged_initial second_catch tagged_in_catch total_fish = 1 / 25 := by
  sorry


end NUMINAMATH_CALUDE_fish_ratio_problem_l3043_304372


namespace NUMINAMATH_CALUDE_number_puzzle_l3043_304338

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 5) = 129 ∧ x = 19 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3043_304338


namespace NUMINAMATH_CALUDE_calculate_expression_l3043_304360

theorem calculate_expression : 10 + 7 * (3 + 8)^2 = 857 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3043_304360


namespace NUMINAMATH_CALUDE_harold_marble_sharing_l3043_304350

theorem harold_marble_sharing (total_marbles : ℕ) (kept_marbles : ℕ) (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 100)
  (h2 : kept_marbles = 20)
  (h3 : marbles_per_friend = 16)
  : (total_marbles - kept_marbles) / marbles_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_harold_marble_sharing_l3043_304350


namespace NUMINAMATH_CALUDE_nadine_chairs_purchase_l3043_304358

/-- Proves that Nadine bought 2 chairs given the conditions of her purchases -/
theorem nadine_chairs_purchase :
  ∀ (total_spent table_cost chair_cost : ℕ),
    total_spent = 56 →
    table_cost = 34 →
    chair_cost = 11 →
    ∃ (num_chairs : ℕ),
      num_chairs * chair_cost = total_spent - table_cost ∧
      num_chairs = 2 := by
  sorry

end NUMINAMATH_CALUDE_nadine_chairs_purchase_l3043_304358


namespace NUMINAMATH_CALUDE_inverse_of_A_l3043_304304

def A : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; -2, 9]

theorem inverse_of_A :
  A⁻¹ = !![9/35, -4/35; 2/35, 3/35] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_l3043_304304


namespace NUMINAMATH_CALUDE_jills_shopping_tax_percentage_l3043_304307

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def totalTaxPercentage (clothingPercent foodPercent otherPercent : ℝ)
                       (clothingTaxRate foodTaxRate otherTaxRate : ℝ) : ℝ :=
  (clothingPercent * clothingTaxRate + foodPercent * foodTaxRate + otherPercent * otherTaxRate) * 100

/-- Theorem stating that the total tax percentage for Jill's shopping trip is 5.20% -/
theorem jills_shopping_tax_percentage :
  totalTaxPercentage 0.50 0.10 0.40 0.04 0 0.08 = 5.20 := by
  sorry

end NUMINAMATH_CALUDE_jills_shopping_tax_percentage_l3043_304307


namespace NUMINAMATH_CALUDE_divisibility_condition_l3043_304319

theorem divisibility_condition (n : ℕ+) :
  (5^(n.val - 1) + 3^(n.val - 1)) ∣ (5^n.val + 3^n.val) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3043_304319


namespace NUMINAMATH_CALUDE_sum_of_numbers_ge_04_l3043_304394

theorem sum_of_numbers_ge_04 : 
  let numbers : List ℚ := [4/5, 1/2, 9/10, 1/3]
  (numbers.filter (λ x => x ≥ 2/5)).sum = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_ge_04_l3043_304394


namespace NUMINAMATH_CALUDE_circular_seating_l3043_304351

theorem circular_seating (total_people : Nat) (seated_people : Nat) (arrangements : Nat) :
  total_people = 6 →
  seated_people ≤ total_people →
  arrangements = 144 →
  arrangements = Nat.factorial (seated_people - 1) →
  seated_people = 5 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_l3043_304351


namespace NUMINAMATH_CALUDE_loss_percent_calculation_l3043_304315

theorem loss_percent_calculation (cost_price selling_price : ℝ) : 
  cost_price = 600 → 
  selling_price = 550 → 
  (cost_price - selling_price) / cost_price * 100 = 8.33 := by
sorry

end NUMINAMATH_CALUDE_loss_percent_calculation_l3043_304315


namespace NUMINAMATH_CALUDE_sin_690_degrees_l3043_304373

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l3043_304373


namespace NUMINAMATH_CALUDE_delightful_numbers_l3043_304396

def is_delightful (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  n % 25 = 0 ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) % 25 = 0 ∧
  ((n / 1000) * (n / 100 % 10) * (n / 10 % 10) * (n % 10)) % 25 = 0

theorem delightful_numbers :
  ∀ n : ℕ, is_delightful n ↔ n = 5875 ∨ n = 8575 := by sorry

end NUMINAMATH_CALUDE_delightful_numbers_l3043_304396


namespace NUMINAMATH_CALUDE_books_checked_out_after_returns_l3043_304352

-- Define the initial state
def initial_books : ℕ := 15
def initial_movies : ℕ := 6

-- Define the number of books returned
def books_returned : ℕ := 8

-- Define the fraction of movies returned
def movie_return_fraction : ℚ := 1 / 3

-- Define the final total of items
def final_total : ℕ := 20

-- Theorem to prove
theorem books_checked_out_after_returns (checked_out : ℕ) : 
  checked_out = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_books_checked_out_after_returns_l3043_304352


namespace NUMINAMATH_CALUDE_work_rate_problem_l3043_304382

/-- Given three workers with work rates satisfying certain conditions,
    prove that two of them together have a specific work rate. -/
theorem work_rate_problem (A B C : ℚ) 
  (h1 : A + B = 1/8)
  (h2 : A + B + C = 1/6)
  (h3 : A + C = 1/8) :
  B + C = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_work_rate_problem_l3043_304382


namespace NUMINAMATH_CALUDE_nine_point_zero_one_closest_l3043_304365

def options : List ℝ := [10.01, 9.998, 9.9, 9.01]

def closest_to_nine (x : ℝ) : Prop :=
  ∀ y ∈ options, |x - 9| ≤ |y - 9|

theorem nine_point_zero_one_closest :
  closest_to_nine 9.01 := by sorry

end NUMINAMATH_CALUDE_nine_point_zero_one_closest_l3043_304365


namespace NUMINAMATH_CALUDE_solution_set_correct_range_of_b_l3043_304344

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + 2

-- Define the solution set for f(x) > 0
def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then Set.Ioo (1/a) 2
  else if a = 0 then Set.Iio 2
  else if 0 < a ∧ a < 1/2 then Set.Iio 2 ∪ Set.Ioi (1/a)
  else if a = 1/2 then Set.Iio 2 ∪ Set.Ioi 2
  else Set.Iio (1/a) ∪ Set.Ioi 2

-- State the theorem for the solution set
theorem solution_set_correct (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ f a x > 0 :=
sorry

-- State the theorem for the range of b
theorem range_of_b :
  ∀ x ∈ Set.Icc (1/3) 1,
  ∀ m ∈ Set.Icc 1 4,
  f 1 (1/x) + (3 - 2*m)/x ≤ b^2 - 2*b - 2 →
  b ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_correct_range_of_b_l3043_304344


namespace NUMINAMATH_CALUDE_equation_solutions_l3043_304339

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (-1 + Real.sqrt 5) / 2 ∧ x₂ = (-1 - Real.sqrt 5) / 2 ∧
    x₁^2 + x₁ - 1 = 0 ∧ x₂^2 + x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 2/3 ∧
    2*(x₁ - 3) = 3*x₁*(x₁ - 3) ∧ 2*(x₂ - 3) = 3*x₂*(x₂ - 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3043_304339


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3043_304337

theorem absolute_value_equation_solutions :
  ∀ x : ℚ, (|2 * x - 3| = x + 1) ↔ (x = 4 ∨ x = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3043_304337


namespace NUMINAMATH_CALUDE_permutations_congruence_l3043_304369

/-- The number of ways to arrange n elements, choosing k of them -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid permutations of "AAAABBBBCCCCDDDD" -/
def N : ℕ :=
  (choose 5 0 * choose 4 4 * choose 3 3 * choose 4 0) +
  (choose 5 1 * choose 4 3 * choose 3 2 * choose 4 1) +
  (choose 5 2 * choose 4 2 * choose 3 1 * choose 4 2) +
  (choose 5 3 * choose 4 1 * choose 3 0 * choose 4 3)

theorem permutations_congruence :
  N ≡ 581 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_permutations_congruence_l3043_304369


namespace NUMINAMATH_CALUDE_total_swordfish_catch_l3043_304345

/-- The number of swordfish Shelly catches per trip -/
def shelly_catch : ℕ := 5 - 2

/-- The number of swordfish Sam catches per trip -/
def sam_catch : ℕ := shelly_catch - 1

/-- The number of fishing trips -/
def num_trips : ℕ := 5

/-- The total number of swordfish caught by Shelly and Sam -/
def total_catch : ℕ := (shelly_catch + sam_catch) * num_trips

theorem total_swordfish_catch : total_catch = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_swordfish_catch_l3043_304345


namespace NUMINAMATH_CALUDE_biking_difference_l3043_304330

/-- Calculates the difference in miles biked between two cyclists given their speeds, 
    total time, and break times. -/
def miles_difference (alberto_speed bjorn_speed total_time alberto_break bjorn_break : ℝ) : ℝ :=
  let alberto_distance := alberto_speed * (total_time - alberto_break)
  let bjorn_distance := bjorn_speed * (total_time - bjorn_break)
  alberto_distance - bjorn_distance

/-- The difference in miles biked between Alberto and Bjorn is 17.625 miles. -/
theorem biking_difference : 
  miles_difference 15 10.5 5 0.5 0.25 = 17.625 := by
  sorry

end NUMINAMATH_CALUDE_biking_difference_l3043_304330
