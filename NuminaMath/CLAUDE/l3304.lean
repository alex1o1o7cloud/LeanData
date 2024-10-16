import Mathlib

namespace NUMINAMATH_CALUDE_largest_trick_number_l3304_330466

/-- The constant k representing the number 2017 -/
def k : ℕ := 2017

/-- A function that determines whether the card trick can be performed for a given number of cards -/
def canPerformTrick (n : ℕ) : Prop :=
  n ≤ k + 1 ∧ (n ≤ k → False)

/-- Theorem stating that 2018 is the largest number for which the trick can be performed -/
theorem largest_trick_number : ∀ n : ℕ, canPerformTrick n ↔ n = k + 1 :=
  sorry

end NUMINAMATH_CALUDE_largest_trick_number_l3304_330466


namespace NUMINAMATH_CALUDE_number_percentage_equality_l3304_330434

theorem number_percentage_equality : ∃ x : ℚ, (3 / 10 : ℚ) * x = (2 / 10 : ℚ) * 40 ∧ x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l3304_330434


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l3304_330488

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) (h2 : x * y = -4) : x^2 + y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l3304_330488


namespace NUMINAMATH_CALUDE_dan_added_sixteen_pencils_l3304_330472

/-- The number of pencils Dan placed on the desk -/
def pencils_added (drawer : ℕ) (desk : ℕ) (total : ℕ) : ℕ :=
  total - (drawer + desk)

/-- Proof that Dan placed 16 pencils on the desk -/
theorem dan_added_sixteen_pencils :
  pencils_added 43 19 78 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dan_added_sixteen_pencils_l3304_330472


namespace NUMINAMATH_CALUDE_trajectory_eq_sufficient_not_necessary_l3304_330455

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The distance from a point to the x-axis -/
def distToXAxis (p : Point2D) : ℝ := |p.y|

/-- The distance from a point to the y-axis -/
def distToYAxis (p : Point2D) : ℝ := |p.x|

/-- A point has equal distance to both axes -/
def equalDistToAxes (p : Point2D) : Prop :=
  distToXAxis p = distToYAxis p

/-- The trajectory equation y = |x| -/
def trajectoryEq (p : Point2D) : Prop :=
  p.y = |p.x|

/-- Theorem: y = |x| is a sufficient but not necessary condition for equal distance to both axes -/
theorem trajectory_eq_sufficient_not_necessary :
  (∀ p : Point2D, trajectoryEq p → equalDistToAxes p) ∧
  (∃ p : Point2D, equalDistToAxes p ∧ ¬trajectoryEq p) :=
sorry

end NUMINAMATH_CALUDE_trajectory_eq_sufficient_not_necessary_l3304_330455


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l3304_330473

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  ((x - 1) * (x + 2)) / (x^2 - 1) = 0 → x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l3304_330473


namespace NUMINAMATH_CALUDE_sp_length_l3304_330496

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8)
  (bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 9)
  (ca_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 10)

-- Define the point T
def T (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the point S
def S (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the point P
def P (triangle : Triangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem sp_length (triangle : Triangle) : 
  Real.sqrt ((S triangle).1 - (P triangle).1)^2 + ((S triangle).2 - (P triangle).2)^2 = 225/13 := by
  sorry

end NUMINAMATH_CALUDE_sp_length_l3304_330496


namespace NUMINAMATH_CALUDE_parking_probability_l3304_330471

/-- Represents a parking lot configuration -/
structure ParkingLot :=
  (total_spaces : ℕ)
  (occupied_spaces : ℕ)

/-- Calculates the probability of finding two adjacent empty spaces in a parking lot -/
def probability_of_two_adjacent_empty_spaces (p : ParkingLot) : ℚ :=
  1 - (Nat.choose (p.total_spaces - p.occupied_spaces + 1) 5 : ℚ) / (Nat.choose p.total_spaces (p.total_spaces - p.occupied_spaces) : ℚ)

/-- Theorem stating the probability of finding two adjacent empty spaces in the given scenario -/
theorem parking_probability (p : ParkingLot) 
  (h1 : p.total_spaces = 20) 
  (h2 : p.occupied_spaces = 15) : 
  probability_of_two_adjacent_empty_spaces p = 232 / 323 := by
  sorry

end NUMINAMATH_CALUDE_parking_probability_l3304_330471


namespace NUMINAMATH_CALUDE_speed_ratio_eddy_freddy_l3304_330449

/-- Proves that the ratio of Eddy's average speed to Freddy's average speed is 38:15 -/
theorem speed_ratio_eddy_freddy : 
  ∀ (eddy_distance freddy_distance : ℝ) 
    (eddy_time freddy_time : ℝ),
  eddy_distance = 570 →
  freddy_distance = 300 →
  eddy_time = 3 →
  freddy_time = 4 →
  (eddy_distance / eddy_time) / (freddy_distance / freddy_time) = 38 / 15 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_eddy_freddy_l3304_330449


namespace NUMINAMATH_CALUDE_M_intersect_N_l3304_330423

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_intersect_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l3304_330423


namespace NUMINAMATH_CALUDE_work_completion_time_l3304_330405

/-- The number of days it takes for a and b together to complete the work -/
def combined_time : ℝ := 6

/-- The number of days it takes for b alone to complete the work -/
def b_time : ℝ := 11.142857142857144

/-- The number of days it takes for a alone to complete the work -/
def a_time : ℝ := 13

/-- The theorem stating that given the combined time and b's time, a's time is 13 days -/
theorem work_completion_time : 
  (1 / combined_time) = (1 / a_time) + (1 / b_time) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3304_330405


namespace NUMINAMATH_CALUDE_compound_interest_proof_l3304_330475

/-- Calculates the final amount after compound interest --/
def final_amount (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that $30,000 increased by 55% annually for 2 years results in $72,075 --/
theorem compound_interest_proof :
  let principal : ℝ := 30000
  let rate : ℝ := 0.55
  let years : ℕ := 2
  final_amount principal rate years = 72075 := by sorry

end NUMINAMATH_CALUDE_compound_interest_proof_l3304_330475


namespace NUMINAMATH_CALUDE_root_value_theorem_l3304_330422

theorem root_value_theorem (a : ℝ) : 
  (2 * a^2 + 3 * a - 4 = 0) → (2 * a^2 + 3 * a = 4) := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l3304_330422


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3304_330429

theorem fraction_equals_zero (x : ℝ) : (x - 5) / (5 * x - 15) = 0 ↔ x = 5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3304_330429


namespace NUMINAMATH_CALUDE_number_satisfies_equation_l3304_330410

theorem number_satisfies_equation : ∃ x : ℝ, (45 - 3 * x^2 = 12) ∧ (x = Real.sqrt 11 ∨ x = -Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_number_satisfies_equation_l3304_330410


namespace NUMINAMATH_CALUDE_plane_not_perp_implies_no_perp_line_l3304_330494

-- Define planes and lines
variable (α β : Set (ℝ × ℝ × ℝ))
variable (l : Set (ℝ × ℝ × ℝ))

-- Define perpendicularity for planes and lines
def perpendicular_planes (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def perpendicular_line_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define a line being in a plane
def line_in_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem plane_not_perp_implies_no_perp_line :
  ¬(perpendicular_planes α β) →
  ¬∃ l, line_in_plane l α ∧ perpendicular_line_plane l β :=
sorry

end NUMINAMATH_CALUDE_plane_not_perp_implies_no_perp_line_l3304_330494


namespace NUMINAMATH_CALUDE_garden_ratio_theorem_l3304_330458

/-- Represents the dimensions of a square garden surrounded by rectangular flower beds -/
structure GardenDimensions where
  s : ℝ  -- side length of the square garden
  x : ℝ  -- longer side of each rectangular bed
  y : ℝ  -- shorter side of each rectangular bed

/-- The theorem stating the ratio of the longer side to the shorter side of each rectangular bed -/
theorem garden_ratio_theorem (d : GardenDimensions) 
  (h1 : d.s > 0)  -- the garden has positive side length
  (h2 : d.s + 2 * d.y = Real.sqrt 3 * d.s)  -- outer square side length relation
  (h3 : d.x + d.y = Real.sqrt 3 * d.s)  -- outer square diagonal relation
  : d.x / d.y = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_theorem_l3304_330458


namespace NUMINAMATH_CALUDE_balcony_difference_l3304_330425

/-- Represents the number of tickets sold for each section of the theater. -/
structure TheaterSales where
  orchestra : ℕ
  balcony : ℕ
  vip : ℕ

/-- Calculates the total revenue from ticket sales. -/
def totalRevenue (sales : TheaterSales) : ℕ :=
  15 * sales.orchestra + 10 * sales.balcony + 20 * sales.vip

/-- Calculates the total number of tickets sold. -/
def totalTickets (sales : TheaterSales) : ℕ :=
  sales.orchestra + sales.balcony + sales.vip

/-- Theorem stating the difference between balcony tickets and the sum of orchestra and VIP tickets. -/
theorem balcony_difference (sales : TheaterSales) 
    (h1 : totalTickets sales = 550)
    (h2 : totalRevenue sales = 8000) :
    sales.balcony - (sales.orchestra + sales.vip) = 370 := by
  sorry

end NUMINAMATH_CALUDE_balcony_difference_l3304_330425


namespace NUMINAMATH_CALUDE_sum_and_equality_condition_l3304_330443

/-- Given three real numbers x, y, and z satisfying the conditions:
    1. x + y + z = 150
    2. (x + 10) = (y - 10) = 3z
    Prove that x = 380/7 -/
theorem sum_and_equality_condition (x y z : ℝ) 
  (sum_eq : x + y + z = 150)
  (equality_cond : (x + 10) = (y - 10) ∧ (x + 10) = 3*z) :
  x = 380/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_equality_condition_l3304_330443


namespace NUMINAMATH_CALUDE_f_two_zeros_sum_greater_than_two_l3304_330485

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (a/2) * x^2 + (a-1) * x

theorem f_two_zeros_sum_greater_than_two (a : ℝ) (h : a > 2) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  f a x₁ = 0 ∧ f a x₂ = 0 ∧
  (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ∧
  x₁ + x₂ > 2 := by sorry

end NUMINAMATH_CALUDE_f_two_zeros_sum_greater_than_two_l3304_330485


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3304_330468

/-- The magnitude of the vector corresponding to the complex number 2/(1+i) is √2 -/
theorem magnitude_of_complex_fraction : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3304_330468


namespace NUMINAMATH_CALUDE_curling_teams_l3304_330498

theorem curling_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_curling_teams_l3304_330498


namespace NUMINAMATH_CALUDE_f_of_tan_squared_l3304_330483

noncomputable def f (x : ℝ) : ℝ := 1 / (((x / (x - 1)) - 1) / (x / (x - 1)))^2

theorem f_of_tan_squared (t : ℝ) (h : 0 ≤ t ∧ t ≤ π/4) :
  f (Real.tan t)^2 = (Real.cos (2*t) / Real.sin t^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_tan_squared_l3304_330483


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3304_330438

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 8 = 1/2 →
  a 6 * (a 2 + 2 * a 6 + a 10) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3304_330438


namespace NUMINAMATH_CALUDE_number_difference_l3304_330463

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21780)
  (a_div_5 : a % 5 = 0)
  (b_relation : b * 10 + 5 = a) :
  a - b = 17825 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l3304_330463


namespace NUMINAMATH_CALUDE_basketball_win_percentage_l3304_330445

theorem basketball_win_percentage (total_games : ℕ) (first_games : ℕ) (first_wins : ℕ) (remaining_games : ℕ) (remaining_wins : ℕ) :
  total_games = first_games + remaining_games →
  first_games = 55 →
  first_wins = 45 →
  remaining_games = 50 →
  remaining_wins = 34 →
  (first_wins + remaining_wins : ℚ) / total_games = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_basketball_win_percentage_l3304_330445


namespace NUMINAMATH_CALUDE_communication_system_probabilities_l3304_330432

/-- Represents a communication system with two signals A and B --/
structure CommunicationSystem where
  pTransmitA : ℝ  -- Probability of transmitting signal A
  pTransmitB : ℝ  -- Probability of transmitting signal B
  pDistortAtoB : ℝ  -- Probability of A being distorted to B
  pDistortBtoA : ℝ  -- Probability of B being distorted to A

/-- Theorem about probabilities in the communication system --/
theorem communication_system_probabilities (sys : CommunicationSystem) 
  (h1 : sys.pTransmitA = 0.72)
  (h2 : sys.pTransmitB = 0.28)
  (h3 : sys.pDistortAtoB = 1/6)
  (h4 : sys.pDistortBtoA = 1/7) :
  let pReceiveA := sys.pTransmitA * (1 - sys.pDistortAtoB) + sys.pTransmitB * sys.pDistortBtoA
  let pTransmittedAGivenReceivedA := (sys.pTransmitA * (1 - sys.pDistortAtoB)) / pReceiveA
  pReceiveA = 0.64 ∧ pTransmittedAGivenReceivedA = 0.9375 := by
  sorry


end NUMINAMATH_CALUDE_communication_system_probabilities_l3304_330432


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_fourth_l3304_330416

theorem r_fourth_plus_inverse_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_fourth_l3304_330416


namespace NUMINAMATH_CALUDE_max_probability_two_color_balls_l3304_330490

def p (n : ℕ+) : ℚ :=
  (10 * n) / ((n + 5) * (n + 4))

theorem max_probability_two_color_balls :
  ∀ n : ℕ+, p n ≤ 5/9 :=
by sorry

end NUMINAMATH_CALUDE_max_probability_two_color_balls_l3304_330490


namespace NUMINAMATH_CALUDE_complex_fraction_power_four_l3304_330404

theorem complex_fraction_power_four (i : ℂ) (h : i * i = -1) : 
  ((1 + i) / (1 - i)) ^ 4 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_power_four_l3304_330404


namespace NUMINAMATH_CALUDE_arithmetic_mean_increase_l3304_330437

theorem arithmetic_mean_increase (a b c d e : ℝ) :
  let original_set := [a, b, c, d, e]
  let new_set := original_set.map (λ x => x + 15)
  let original_mean := (a + b + c + d + e) / 5
  let new_mean := (new_set.sum) / 5
  new_mean = original_mean + 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_increase_l3304_330437


namespace NUMINAMATH_CALUDE_set_of_naturals_less_than_three_l3304_330489

theorem set_of_naturals_less_than_three :
  {x : ℕ | x < 3} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_of_naturals_less_than_three_l3304_330489


namespace NUMINAMATH_CALUDE_remainder_7645_div_9_l3304_330499

theorem remainder_7645_div_9 : 7645 % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7645_div_9_l3304_330499


namespace NUMINAMATH_CALUDE_exactly_two_statements_true_l3304_330442

theorem exactly_two_statements_true :
  let statement1 := (¬∀ x : ℝ, x^2 - 3*x - 2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 3*x₀ - 2 ≤ 0)
  let statement2 := ∀ P Q : Prop, (P ∨ Q → P ∧ Q) ∧ ¬(P ∧ Q → P ∨ Q)
  let statement3 := ∃ m : ℝ, ∀ x : ℝ, x > 0 → (
    (∃ α : ℝ, ∀ x : ℝ, x > 0 → m * x^(m^2 + 2*m) = x^α) ∧
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → m * x₁^(m^2 + 2*m) < m * x₂^(m^2 + 2*m))
  )
  let statement4 := ∀ a b : ℝ, a ≠ 0 ∧ b ≠ 0 →
    (∀ x y : ℝ, x/a + y/b = 1 ↔ ∃ k : ℝ, k ≠ 0 ∧ x = k*a ∧ y = k*b)
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_statements_true_l3304_330442


namespace NUMINAMATH_CALUDE_total_nuts_equals_1_05_l3304_330433

/-- The amount of walnuts Karen added to the trail mix in cups -/
def w : ℝ := 0.25

/-- The amount of almonds Karen added to the trail mix in cups -/
def a : ℝ := 0.25

/-- The amount of peanuts Karen added to the trail mix in cups -/
def p : ℝ := 0.15

/-- The amount of cashews Karen added to the trail mix in cups -/
def c : ℝ := 0.40

/-- The total amount of nuts Karen added to the trail mix -/
def total_nuts : ℝ := w + a + p + c

theorem total_nuts_equals_1_05 : total_nuts = 1.05 := by sorry

end NUMINAMATH_CALUDE_total_nuts_equals_1_05_l3304_330433


namespace NUMINAMATH_CALUDE_equation_solutions_l3304_330482

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 = 5 * x₁ ∧ x₁ = 0) ∧ (2 * x₂^2 = 5 * x₂ ∧ x₂ = 5/2)) ∧
  (∃ y₁ y₂ : ℝ, (y₁^2 + 3*y₁ = 3 ∧ y₁ = (-3 + Real.sqrt 21) / 2) ∧
               (y₂^2 + 3*y₂ = 3 ∧ y₂ = (-3 - Real.sqrt 21) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3304_330482


namespace NUMINAMATH_CALUDE_distance_on_line_l3304_330406

/-- The distance between two points on a line y = kx + b -/
theorem distance_on_line (k b x₁ x₂ : ℝ) :
  let P : ℝ × ℝ := (x₁, k * x₁ + b)
  let Q : ℝ × ℝ := (x₂, k * x₂ + b)
  ‖P - Q‖ = |x₁ - x₂| * Real.sqrt (1 + k^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_on_line_l3304_330406


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3304_330464

theorem quadratic_equation_roots (a : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*a*x₁ + a^2 - 4 = 0) ∧ 
  (x₂^2 - 2*a*x₂ + a^2 - 4 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3304_330464


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3304_330467

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 4 / 5 →  -- ratio of angles is 4:5
  |a - b| = 10 :=  -- positive difference is 10°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3304_330467


namespace NUMINAMATH_CALUDE_exists_same_color_distance_exists_color_for_all_distances_l3304_330424

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorAssignment : Point → Color := sorry

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Statement for part (i)
theorem exists_same_color_distance (x : ℝ) :
  ∃ (c : Color) (p1 p2 : Point),
    colorAssignment p1 = c ∧
    colorAssignment p2 = c ∧
    distance p1 p2 = x :=
  sorry

-- Statement for part (ii)
theorem exists_color_for_all_distances :
  ∃ (c : Color), ∀ (x : ℝ),
    ∃ (p1 p2 : Point),
      colorAssignment p1 = c ∧
      colorAssignment p2 = c ∧
      distance p1 p2 = x :=
  sorry

end NUMINAMATH_CALUDE_exists_same_color_distance_exists_color_for_all_distances_l3304_330424


namespace NUMINAMATH_CALUDE_johnny_table_legs_l3304_330414

/-- Given the number of tables, planks per surface, and total planks,
    calculate the number of planks needed for the legs of each table. -/
def planksForLegs (numTables : ℕ) (planksPerSurface : ℕ) (totalPlanks : ℕ) : ℕ :=
  (totalPlanks - numTables * planksPerSurface) / numTables

/-- Theorem stating that given the specific values in the problem,
    the number of planks needed for the legs of each table is 4. -/
theorem johnny_table_legs :
  planksForLegs 5 5 45 = 4 := by
  sorry

end NUMINAMATH_CALUDE_johnny_table_legs_l3304_330414


namespace NUMINAMATH_CALUDE_c_share_of_profit_l3304_330411

/-- Calculates the share of profit for a partner in a business partnership. -/
def calculate_share (investment : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

/-- Proves that C's share of profit is 36,000 given the specified investments and total profit. -/
theorem c_share_of_profit (a b c total_profit : ℕ) 
  (ha : a = 24000)
  (hb : b = 32000)
  (hc : c = 36000)
  (htotal : total_profit = 92000) :
  calculate_share c (a + b + c) total_profit = 36000 := by
sorry

#eval calculate_share 36000 92000 92000

end NUMINAMATH_CALUDE_c_share_of_profit_l3304_330411


namespace NUMINAMATH_CALUDE_abs_neg_three_l3304_330403

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_l3304_330403


namespace NUMINAMATH_CALUDE_max_value_product_l3304_330480

theorem max_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5*x + 2*y < 50) :
  xy*(50 - 5*x - 2*y) ≤ 125000/432 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 5*x₀ + 2*y₀ < 50 ∧ x₀*y₀*(50 - 5*x₀ - 2*y₀) = 125000/432 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l3304_330480


namespace NUMINAMATH_CALUDE_distribute_six_tasks_three_people_l3304_330452

/-- The number of ways to distribute tasks among people -/
def distribute_tasks (num_tasks : ℕ) (num_people : ℕ) : ℕ :=
  num_people^num_tasks - num_people * (num_people - 1)^num_tasks + num_people

/-- Theorem stating the correct number of ways to distribute 6 tasks among 3 people -/
theorem distribute_six_tasks_three_people :
  distribute_tasks 6 3 = 540 := by
  sorry


end NUMINAMATH_CALUDE_distribute_six_tasks_three_people_l3304_330452


namespace NUMINAMATH_CALUDE_subtract_negatives_l3304_330454

theorem subtract_negatives : (-1) - (-4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_l3304_330454


namespace NUMINAMATH_CALUDE_reduced_banana_price_l3304_330446

/-- Given a 60% reduction in banana prices and the ability to obtain 120 more bananas
    for Rs. 150 after the reduction, prove that the reduced price per dozen bananas
    is Rs. 48/17. -/
theorem reduced_banana_price (P : ℚ) : 
  (150 / (0.4 * P) = 150 / P + 120) →
  (12 * (0.4 * P) = 48 / 17) :=
by sorry

end NUMINAMATH_CALUDE_reduced_banana_price_l3304_330446


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3304_330436

theorem container_volume_ratio : 
  ∀ (C D : ℚ), C > 0 → D > 0 → 
  (3 / 4 : ℚ) * C = (5 / 8 : ℚ) * D → 
  C / D = (5 / 6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3304_330436


namespace NUMINAMATH_CALUDE_eighth_of_2_36_equals_2_33_l3304_330402

theorem eighth_of_2_36_equals_2_33 : ∃ y : ℕ, (1 / 8 : ℝ) * (2 ^ 36) = 2 ^ y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_eighth_of_2_36_equals_2_33_l3304_330402


namespace NUMINAMATH_CALUDE_tennis_balls_count_l3304_330493

theorem tennis_balls_count (baskets : ℕ) (soccer_balls : ℕ) (students_8 : ℕ) (students_10 : ℕ) 
  (balls_removed_8 : ℕ) (balls_removed_10 : ℕ) (balls_remaining : ℕ) :
  baskets = 5 →
  soccer_balls = 5 →
  students_8 = 3 →
  students_10 = 2 →
  balls_removed_8 = 8 →
  balls_removed_10 = 10 →
  balls_remaining = 56 →
  ∃ T : ℕ, 
    baskets * (T + soccer_balls) - (students_8 * balls_removed_8 + students_10 * balls_removed_10) = balls_remaining ∧
    T = 15 :=
by sorry

end NUMINAMATH_CALUDE_tennis_balls_count_l3304_330493


namespace NUMINAMATH_CALUDE_minimize_constant_term_l3304_330439

/-- The function representing the constant term in the expansion -/
def f (a : ℝ) : ℝ := a^3 - 9*a

/-- Theorem stating that √3 minimizes f(a) for a > 0 -/
theorem minimize_constant_term (a : ℝ) (h : a > 0) :
  f a ≥ f (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_minimize_constant_term_l3304_330439


namespace NUMINAMATH_CALUDE_common_divisors_9240_8000_l3304_330474

theorem common_divisors_9240_8000 : ∃ n : ℕ, n = (Nat.divisors (Nat.gcd 9240 8000)).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_8000_l3304_330474


namespace NUMINAMATH_CALUDE_count_valid_triangles_l3304_330401

/-- A point in the 4x4 grid --/
structure GridPoint where
  x : Fin 4
  y : Fin 4

/-- A triangle formed by three grid points --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Function to check if three points are collinear --/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Function to check if a triangle has positive area --/
def positiveArea (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all possible grid points --/
def allGridPoints : Finset GridPoint :=
  sorry

/-- The set of all possible triangles with positive area --/
def validTriangles : Finset GridTriangle :=
  sorry

/-- The main theorem --/
theorem count_valid_triangles :
  Finset.card validTriangles = 516 :=
sorry

end NUMINAMATH_CALUDE_count_valid_triangles_l3304_330401


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l3304_330469

def A (p : ℝ) : Set ℝ := {x | 2 * x^2 + 3 * p * x + 2 = 0}
def B (q : ℝ) : Set ℝ := {x | 2 * x^2 + x + q = 0}

theorem intersection_and_union_of_sets (p q : ℝ) :
  A p ∩ B q = {1/2} →
  p = -5/3 ∧ q = -1 ∧ A p ∪ B q = {-1, 1/2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l3304_330469


namespace NUMINAMATH_CALUDE_bills_omelet_time_l3304_330453

/-- The time it takes to prepare and cook omelets -/
def total_time (pepper_chop_time onion_chop_time cheese_grate_time assemble_cook_time : ℕ) 
               (num_peppers num_onions num_omelets : ℕ) : ℕ :=
  (pepper_chop_time * num_peppers) + 
  (onion_chop_time * num_onions) + 
  ((cheese_grate_time + assemble_cook_time) * num_omelets)

/-- Theorem stating that Bill's total preparation and cooking time for five omelets is 50 minutes -/
theorem bills_omelet_time : 
  total_time 3 4 1 5 4 2 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bills_omelet_time_l3304_330453


namespace NUMINAMATH_CALUDE_first_group_size_correct_l3304_330447

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 33

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group works -/
def second_group_days : ℕ := 11

/-- The number of hours per day the second group works -/
def second_group_hours : ℕ := 6

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_correct :
  first_group_size * first_group_hours * first_group_days =
  second_group_size * second_group_hours * second_group_days :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l3304_330447


namespace NUMINAMATH_CALUDE_circle_radius_problem_l3304_330435

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def collinear (a b c : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b = a + t • (c - a)

def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

def share_common_tangent (c1 c2 c3 : Circle) : Prop :=
  ∃ (l : ℝ × ℝ → Prop), ∀ (p : ℝ × ℝ),
    l p → (∃ (q : ℝ × ℝ), l q ∧ 
      ((p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∨
       (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 ∨
       (p.1 - c3.center.1)^2 + (p.2 - c3.center.2)^2 = c3.radius^2))

theorem circle_radius_problem (A B C : Circle) 
  (h1 : collinear A.center B.center C.center)
  (h2 : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ B.center = A.center + t • (C.center - A.center))
  (h3 : externally_tangent A B)
  (h4 : externally_tangent B C)
  (h5 : share_common_tangent A B C)
  (h6 : A.radius = 12)
  (h7 : B.radius = 42) :
  C.radius = 147 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l3304_330435


namespace NUMINAMATH_CALUDE_abs_one_fifth_set_l3304_330457

theorem abs_one_fifth_set : 
  {x : ℝ | |x| = (1 : ℝ) / 5} = {-(1 : ℝ) / 5, (1 : ℝ) / 5} := by
  sorry

end NUMINAMATH_CALUDE_abs_one_fifth_set_l3304_330457


namespace NUMINAMATH_CALUDE_earliest_meet_time_proof_l3304_330428

def charlie_lap_time : ℕ := 5
def alex_lap_time : ℕ := 8
def taylor_lap_time : ℕ := 10

def earliest_meet_time : ℕ := 40

theorem earliest_meet_time_proof :
  lcm (lcm charlie_lap_time alex_lap_time) taylor_lap_time = earliest_meet_time :=
by sorry

end NUMINAMATH_CALUDE_earliest_meet_time_proof_l3304_330428


namespace NUMINAMATH_CALUDE_bridget_middle_score_l3304_330495

/-- Represents the test scores of the four students -/
structure Scores where
  hannah : ℝ
  ella : ℝ
  cassie : ℝ
  bridget : ℝ

/-- Defines the conditions given in the problem -/
def SatisfiesConditions (s : Scores) : Prop :=
  (s.cassie > s.hannah) ∧ (s.cassie > s.ella) ∧
  (s.bridget ≥ s.hannah) ∧ (s.bridget ≥ s.ella)

/-- Defines what it means for a student to have the middle score -/
def HasMiddleScore (name : String) (s : Scores) : Prop :=
  match name with
  | "Bridget" => (s.bridget > min s.hannah s.ella) ∧ (s.bridget < max s.cassie s.ella)
  | _ => False

/-- The main theorem stating that if the conditions are satisfied, Bridget must have the middle score -/
theorem bridget_middle_score (s : Scores) :
  SatisfiesConditions s → HasMiddleScore "Bridget" s := by
  sorry


end NUMINAMATH_CALUDE_bridget_middle_score_l3304_330495


namespace NUMINAMATH_CALUDE_not_perfect_square_l3304_330460

theorem not_perfect_square (n : ℕ) : ¬∃ (m : ℕ), 2 * 13^n + 5 * 7^n + 26 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3304_330460


namespace NUMINAMATH_CALUDE_lollipop_cost_theorem_lollipop_cost_correct_l3304_330450

/-- The cost of n lollipops given that 2 lollipops cost $2.40 and 6 lollipops cost $7.20 -/
def lollipop_cost (n : ℕ) : ℚ :=
  1.20 * n

/-- Theorem stating that the lollipop_cost function satisfies the given conditions -/
theorem lollipop_cost_theorem :
  lollipop_cost 2 = 2.40 ∧ lollipop_cost 6 = 7.20 :=
by sorry

/-- Theorem proving that the lollipop_cost function is correct for all non-negative integers -/
theorem lollipop_cost_correct (n : ℕ) :
  lollipop_cost n = 1.20 * n :=
by sorry

end NUMINAMATH_CALUDE_lollipop_cost_theorem_lollipop_cost_correct_l3304_330450


namespace NUMINAMATH_CALUDE_quadruple_theorem_l3304_330448

def is_valid_quadruple (a b c d : ℝ) : Prop :=
  (a = b * c ∨ a = b * d ∨ a = c * d) ∧
  (b = a * c ∨ b = a * d ∨ b = c * d) ∧
  (c = a * b ∨ c = a * d ∨ c = b * d) ∧
  (d = a * b ∨ d = a * c ∨ d = b * c)

def is_solution_quadruple (a b c d : ℝ) : Prop :=
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  ((a = 1 ∧ b = 1 ∧ c = -1 ∧ d = -1) ∨
   (a = 1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
   (a = 1 ∧ b = -1 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = 1 ∧ d = -1) ∨
   (a = -1 ∧ b = 1 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = -1 ∧ c = 1 ∧ d = 1)) ∨
  ((a = 1 ∧ b = -1 ∧ c = -1 ∧ d = -1) ∨
   (a = -1 ∧ b = 1 ∧ c = -1 ∧ d = -1) ∨
   (a = -1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
   (a = -1 ∧ b = -1 ∧ c = -1 ∧ d = 1))

theorem quadruple_theorem (a b c d : ℝ) :
  is_valid_quadruple a b c d → is_solution_quadruple a b c d := by
  sorry

end NUMINAMATH_CALUDE_quadruple_theorem_l3304_330448


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_factorization_exists_smallest_b_is_85_l3304_330491

theorem smallest_factorization_coefficient (b : ℕ) : 
  (∃ r s : ℤ, x^2 + b*x + 1800 = (x + r) * (x + s)) →
  b ≥ 85 :=
by sorry

theorem factorization_exists : 
  ∃ r s : ℤ, x^2 + 85*x + 1800 = (x + r) * (x + s) :=
by sorry

theorem smallest_b_is_85 : 
  (∃ r s : ℤ, x^2 + 85*x + 1800 = (x + r) * (x + s)) ∧
  (∀ b : ℕ, b < 85 → ¬(∃ r s : ℤ, x^2 + b*x + 1800 = (x + r) * (x + s))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_factorization_exists_smallest_b_is_85_l3304_330491


namespace NUMINAMATH_CALUDE_students_speaking_both_languages_l3304_330451

/-- Theorem: In a class of 150 students, given that 55 speak English, 85 speak Telugu, 
    and 30 speak neither English nor Telugu, prove that 20 students speak both English and Telugu. -/
theorem students_speaking_both_languages (total : ℕ) (english : ℕ) (telugu : ℕ) (neither : ℕ) :
  total = 150 →
  english = 55 →
  telugu = 85 →
  neither = 30 →
  english + telugu - (total - neither) = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_students_speaking_both_languages_l3304_330451


namespace NUMINAMATH_CALUDE_bianca_drawing_time_l3304_330419

/-- The number of minutes Bianca spent drawing at school -/
def minutes_at_school : ℕ := sorry

/-- The number of minutes Bianca spent drawing at home -/
def minutes_at_home : ℕ := 19

/-- The total number of minutes Bianca spent drawing -/
def total_minutes : ℕ := 41

/-- Theorem stating that Bianca spent 22 minutes drawing at school -/
theorem bianca_drawing_time : minutes_at_school = 22 := by
  sorry

end NUMINAMATH_CALUDE_bianca_drawing_time_l3304_330419


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3304_330431

/-- The sum of an arithmetic sequence with first term 1, common difference 2, and 20 terms -/
def arithmetic_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => (n + 1) + arithmetic_sum n

/-- The first term of the sequence -/
def a₁ : ℕ := 1

/-- The common difference of the sequence -/
def d : ℕ := 2

/-- The number of terms in the sequence -/
def n : ℕ := 20

/-- The n-th term of the sequence -/
def aₙ (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_sum :
  arithmetic_sum n = n * (a₁ + aₙ n) / 2 ∧ arithmetic_sum n = 400 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3304_330431


namespace NUMINAMATH_CALUDE_nissan_cars_sold_l3304_330418

theorem nissan_cars_sold (total_cars : ℕ) (audi_percent : ℚ) (toyota_percent : ℚ) (acura_percent : ℚ) (bmw_percent : ℚ) 
  (h1 : total_cars = 250)
  (h2 : audi_percent = 10 / 100)
  (h3 : toyota_percent = 25 / 100)
  (h4 : acura_percent = 15 / 100)
  (h5 : bmw_percent = 18 / 100)
  : ℕ :=
by
  sorry

#check nissan_cars_sold

end NUMINAMATH_CALUDE_nissan_cars_sold_l3304_330418


namespace NUMINAMATH_CALUDE_youtube_video_dislikes_l3304_330465

theorem youtube_video_dislikes (initial_likes : ℕ) (initial_dislikes : ℕ) (additional_dislikes : ℕ) : 
  initial_likes = 3000 →
  initial_dislikes = initial_likes / 2 + 100 →
  additional_dislikes = 1000 →
  initial_dislikes + additional_dislikes = 2600 :=
by
  sorry

end NUMINAMATH_CALUDE_youtube_video_dislikes_l3304_330465


namespace NUMINAMATH_CALUDE_factorization_proof_l3304_330421

theorem factorization_proof (a m n : ℝ) : a * m^2 - 2 * a * m * n + a * n^2 = a * (m - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3304_330421


namespace NUMINAMATH_CALUDE_right_handed_players_count_l3304_330427

/-- Represents a football team with various player categories -/
structure FootballTeam where
  total_players : ℕ
  thrower_percentage : ℚ
  kicker_percentage : ℚ
  left_handed_remaining_percentage : ℚ
  left_handed_kicker_percentage : ℚ
  exclusive_thrower_percentage : ℚ

/-- Calculates the number of right-handed players and exclusive throwers -/
def calculate_right_handed_players (team : FootballTeam) : ℕ × ℕ :=
  sorry

/-- Theorem stating the correct number of right-handed players and exclusive throwers -/
theorem right_handed_players_count (team : FootballTeam) 
  (h1 : team.total_players = 180)
  (h2 : team.thrower_percentage = 3/10)
  (h3 : team.kicker_percentage = 9/40)
  (h4 : team.left_handed_remaining_percentage = 3/7)
  (h5 : team.left_handed_kicker_percentage = 1/4)
  (h6 : team.exclusive_thrower_percentage = 3/5) :
  calculate_right_handed_players team = (134, 32) := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l3304_330427


namespace NUMINAMATH_CALUDE_infinite_perfect_squares_in_sequence_l3304_330492

theorem infinite_perfect_squares_in_sequence : 
  ∀ k : ℕ, ∃ n : ℕ, ∃ m : ℕ, 2^n + 4^k = m^2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_perfect_squares_in_sequence_l3304_330492


namespace NUMINAMATH_CALUDE_plane_curve_mass_approx_l3304_330441

noncomputable def curve_mass (a b : Real) : Real :=
  ∫ x in a..b, (1 + x^2) * Real.sqrt (1 + (3 * x^2)^2)

theorem plane_curve_mass_approx : 
  ∃ ε > 0, abs (curve_mass 0 0.1 - 0.099985655) < ε :=
sorry

end NUMINAMATH_CALUDE_plane_curve_mass_approx_l3304_330441


namespace NUMINAMATH_CALUDE_product_of_extreme_roots_l3304_330476

-- Define the equation
def equation (x : ℝ) : Prop := x * |x| - 5 * |x| + 6 = 0

-- Define the set of roots
def roots : Set ℝ := {x : ℝ | equation x}

-- Statement to prove
theorem product_of_extreme_roots :
  ∃ (max_root min_root : ℝ),
    max_root ∈ roots ∧
    min_root ∈ roots ∧
    (∀ x ∈ roots, x ≤ max_root) ∧
    (∀ x ∈ roots, x ≥ min_root) ∧
    max_root * min_root = -3 :=
sorry

end NUMINAMATH_CALUDE_product_of_extreme_roots_l3304_330476


namespace NUMINAMATH_CALUDE_left_square_side_length_l3304_330440

/-- Proves that given three squares with specific side length relationships, 
    the left square has a side length of 8 cm. -/
theorem left_square_side_length : 
  ∀ (left middle right : ℝ),
  left + middle + right = 52 →
  middle = left + 17 →
  right = middle - 6 →
  left = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_left_square_side_length_l3304_330440


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_plane_l3304_330478

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane in three-dimensional space -/
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

theorem symmetric_point_xoz_plane :
  let A : Point3D := ⟨-1, 2, 3⟩
  let Q : Point3D := symmetricPointXOZ A
  Q = ⟨-1, -2, 3⟩ := by sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_plane_l3304_330478


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l3304_330462

/-- Given a rectangle with length 4x inches and width 3x + 4 inches,
    where its area is twice its perimeter, prove that x = 1. -/
theorem rectangle_area_perimeter_relation (x : ℝ) : 
  (4 * x) * (3 * x + 4) = 2 * (2 * (4 * x) + 2 * (3 * x + 4)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l3304_330462


namespace NUMINAMATH_CALUDE_ben_win_probability_l3304_330477

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5 / 8) 
  (h2 : ¬ ∃ (draw_prob : ℚ), draw_prob ≠ 0) : 
  1 - lose_prob = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_ben_win_probability_l3304_330477


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3304_330484

/-- 
Given two points A and B in the Cartesian coordinate system,
where A has coordinates (2, m) and B has coordinates (n, -1),
if A and B are symmetric with respect to the x-axis,
then m + n = 3.
-/
theorem symmetric_points_sum (m n : ℝ) : 
  (2 : ℝ) = n ∧ m = -(-1 : ℝ) → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3304_330484


namespace NUMINAMATH_CALUDE_symmetric_point_of_P_l3304_330412

/-- The line y = x + 1 -/
def line (x : ℝ) : ℝ := x + 1

/-- The original point P -/
def P : ℝ × ℝ := (-2, 1)

/-- The symmetric point Q -/
def Q : ℝ × ℝ := (0, -1)

/-- Checks if two points are symmetric with respect to the given line -/
def is_symmetric (p q : ℝ × ℝ) : Prop :=
  let midpoint := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
  midpoint.2 = line midpoint.1

theorem symmetric_point_of_P :
  is_symmetric P Q :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_of_P_l3304_330412


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l3304_330426

theorem cube_sum_reciprocal (x : ℝ) (h : x ≠ 0) :
  x + 1 / x = 3 → x^3 + 1 / x^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l3304_330426


namespace NUMINAMATH_CALUDE_prob_12th_roll_last_l3304_330461

/-- The number of sides on the die -/
def n : ℕ := 8

/-- The number of rolls -/
def r : ℕ := 12

/-- The probability of rolling the same number on the rth roll as on the (r-1)th roll -/
def p_same : ℚ := 1 / n

/-- The probability of rolling a different number on the rth roll from the (r-1)th roll -/
def p_diff : ℚ := (n - 1) / n

/-- The probability that the rth roll is the last roll in the sequence -/
def prob_last_roll (r : ℕ) : ℚ := p_diff^(r - 2) * p_same

theorem prob_12th_roll_last :
  prob_last_roll r = (n - 1)^(r - 2) / n^r := by sorry

end NUMINAMATH_CALUDE_prob_12th_roll_last_l3304_330461


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3304_330407

theorem sum_of_roots_quadratic (m n : ℝ) : 
  (m^2 + 2*m - 1 = 0) → (n^2 + 2*n - 1 = 0) → (m + n = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3304_330407


namespace NUMINAMATH_CALUDE_correct_equation_by_moving_digit_l3304_330415

theorem correct_equation_by_moving_digit : ∃ (a b c : ℕ), 
  (a = 101 ∧ b = 10 ∧ c = 2) ∧ 
  (a - b^c = 1) ∧
  (∃ (x y : ℕ), x * 10 + y = 102 ∧ (x = 10 ∧ y = c)) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_equation_by_moving_digit_l3304_330415


namespace NUMINAMATH_CALUDE_new_person_age_l3304_330417

/-- Given a group of 10 people, prove that if replacing a 44-year-old person
    with a new person decreases the average age by 3 years, then the age of
    the new person is 14 years. -/
theorem new_person_age (group_size : ℕ) (old_person_age : ℕ) (avg_decrease : ℕ) :
  group_size = 10 →
  old_person_age = 44 →
  avg_decrease = 3 →
  ∃ (new_person_age : ℕ),
    (group_size * (avg_decrease + new_person_age) : ℤ) = old_person_age - new_person_age ∧
    new_person_age = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l3304_330417


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3304_330413

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (3 : ℚ) / 5 < p' / q' ∧ p' / q' < (5 : ℚ) / 8 → q ≤ q') →
  q - p = 5 := by sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3304_330413


namespace NUMINAMATH_CALUDE_tuesday_temperature_l3304_330456

theorem tuesday_temperature
  (temp_tue wed thu fri : ℝ)
  (h1 : (temp_tue + wed + thu) / 3 = 45)
  (h2 : (wed + thu + fri) / 3 = 50)
  (h3 : fri = 53) :
  temp_tue = 38 := by
sorry

end NUMINAMATH_CALUDE_tuesday_temperature_l3304_330456


namespace NUMINAMATH_CALUDE_distance_to_origin_l3304_330400

/-- The distance from point P(1, 2, 2) to the origin (0, 0, 0) is 3. -/
theorem distance_to_origin : Real.sqrt (1^2 + 2^2 + 2^2) = 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_origin_l3304_330400


namespace NUMINAMATH_CALUDE_pet_store_cages_l3304_330409

def bird_cages (total_birds : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ) : ℕ :=
  total_birds / (parrots_per_cage + parakeets_per_cage)

theorem pet_store_cages :
  bird_cages 36 2 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3304_330409


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3304_330408

theorem complex_equation_solutions :
  ∃ (S : Set ℂ), S = {z : ℂ | z^6 + 6*I = 0} ∧
  S = {I, -I} ∪ {z : ℂ | ∃ k : ℕ, 0 ≤ k ∧ k < 4 ∧ z = (-6*I)^(1/6) * (Complex.exp (2*π*I*(k:ℝ)/4))} :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3304_330408


namespace NUMINAMATH_CALUDE_fraction_problem_l3304_330444

theorem fraction_problem (numerator : ℕ) : 
  (numerator : ℚ) / (2 * numerator + 4) = 3 / 7 → numerator = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3304_330444


namespace NUMINAMATH_CALUDE_line_b_production_l3304_330486

/-- Given three production lines A, B, and C forming an arithmetic sequence,
    prove that Line B produced 4400 units out of a total of 13200 units. -/
theorem line_b_production (total : ℕ) (a b c : ℕ) : 
  total = 13200 →
  a + b + c = total →
  ∃ (d : ℤ), a = b - d ∧ c = b + d →
  b = 4400 := by
  sorry

end NUMINAMATH_CALUDE_line_b_production_l3304_330486


namespace NUMINAMATH_CALUDE_seven_b_equals_ten_l3304_330430

theorem seven_b_equals_ten (a b : ℚ) (h1 : 5 * a + 2 * b = 0) (h2 : b - 2 = a) : 7 * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_seven_b_equals_ten_l3304_330430


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_l3304_330479

/-- The expected worth of an unfair coin flip -/
theorem expected_worth_unfair_coin : 
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℤ := 5
  let loss_tails : ℤ := 6
  p_heads * gain_heads - p_tails * loss_tails = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_l3304_330479


namespace NUMINAMATH_CALUDE_age_difference_constant_l3304_330459

theorem age_difference_constant (seokjin_initial_age mother_initial_age years_passed : ℕ) :
  mother_initial_age - seokjin_initial_age = 
  (mother_initial_age + years_passed) - (seokjin_initial_age + years_passed) :=
by sorry

end NUMINAMATH_CALUDE_age_difference_constant_l3304_330459


namespace NUMINAMATH_CALUDE_min_value_expression_l3304_330420

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3304_330420


namespace NUMINAMATH_CALUDE_lisa_patricia_ratio_l3304_330487

/-- Represents the money each person has -/
structure Money where
  patricia : ℕ
  lisa : ℕ
  charlotte : ℕ

/-- The conditions of the baseball card problem -/
def baseball_card_problem (m : Money) : Prop :=
  m.patricia = 6 ∧
  m.lisa = 2 * m.charlotte ∧
  m.patricia + m.lisa + m.charlotte = 51

theorem lisa_patricia_ratio (m : Money) :
  baseball_card_problem m →
  m.lisa / m.patricia = 5 := by
  sorry

end NUMINAMATH_CALUDE_lisa_patricia_ratio_l3304_330487


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_120_3507_l3304_330481

theorem gcd_lcm_sum_120_3507 : 
  Nat.gcd 120 3507 + Nat.lcm 120 3507 = 140283 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_120_3507_l3304_330481


namespace NUMINAMATH_CALUDE_count_pairs_eq_50_l3304_330470

/-- The number of pairs of positive integers (m,n) satisfying m^2 + mn < 30 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.1 * p.2 < 30) (Finset.product (Finset.range 30) (Finset.range 30))).card

/-- Theorem stating that the count of pairs satisfying the condition is 50 -/
theorem count_pairs_eq_50 : count_pairs = 50 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_50_l3304_330470


namespace NUMINAMATH_CALUDE_decimal_place_150_l3304_330497

/-- The decimal representation of 5/6 -/
def decimal_rep_5_6 : ℚ := 5/6

/-- The length of the repeating cycle in the decimal representation of 5/6 -/
def cycle_length : ℕ := 6

/-- The nth digit in the decimal representation of 5/6 -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem decimal_place_150 :
  nth_digit 150 = 3 :=
sorry

end NUMINAMATH_CALUDE_decimal_place_150_l3304_330497
