import Mathlib

namespace instantaneous_velocity_at_3_l3055_305505

/-- The displacement function for the object's motion -/
def displacement (t : ℝ) : ℝ := 2 * t^3

/-- The velocity function, which is the derivative of the displacement function -/
def velocity (t : ℝ) : ℝ := 6 * t^2

theorem instantaneous_velocity_at_3 : velocity 3 = 54 := by
  sorry

end instantaneous_velocity_at_3_l3055_305505


namespace distance_to_school_l3055_305508

def walking_speed : ℝ := 80
def travel_time : ℝ := 28

theorem distance_to_school :
  walking_speed * travel_time = 2240 := by sorry

end distance_to_school_l3055_305508


namespace womens_average_age_l3055_305510

theorem womens_average_age (n : ℕ) (A : ℝ) (W₁ W₂ : ℝ) : 
  n = 6 ∧ 
  n * A - 10 - 12 + W₁ + W₂ = n * (A + 2) → 
  (W₁ + W₂) / 2 = 17 := by
  sorry

end womens_average_age_l3055_305510


namespace binomial_12_4_l3055_305594

theorem binomial_12_4 : Nat.choose 12 4 = 495 := by
  sorry

end binomial_12_4_l3055_305594


namespace arithmetic_sequence_problem_l3055_305585

theorem arithmetic_sequence_problem (k : ℕ+) : 
  let a : ℕ → ℤ := λ n => 2*n + 2
  let S : ℕ → ℤ := λ n => n^2 + 3*n
  S k - a (k + 5) = 44 → k = 7 := by
sorry

end arithmetic_sequence_problem_l3055_305585


namespace rhombus_area_l3055_305577

/-- The area of a rhombus with sides of length 4 and an acute angle of 45 degrees is 16 square units -/
theorem rhombus_area (side_length : ℝ) (acute_angle : ℝ) : 
  side_length = 4 → 
  acute_angle = 45 * π / 180 →
  side_length * side_length * Real.sin acute_angle = 16 := by
  sorry

end rhombus_area_l3055_305577


namespace dinitrogen_monoxide_weight_is_44_02_l3055_305560

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in dinitrogen monoxide -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in dinitrogen monoxide -/
def oxygen_count : ℕ := 1

/-- The molecular weight of dinitrogen monoxide (N2O) in g/mol -/
def dinitrogen_monoxide_weight : ℝ :=
  nitrogen_count * nitrogen_weight + oxygen_count * oxygen_weight

/-- Theorem stating that the molecular weight of dinitrogen monoxide is 44.02 g/mol -/
theorem dinitrogen_monoxide_weight_is_44_02 :
  dinitrogen_monoxide_weight = 44.02 := by
  sorry

end dinitrogen_monoxide_weight_is_44_02_l3055_305560


namespace stable_journey_population_l3055_305504

/-- Represents the interstellar vehicle Gibraltar --/
structure Gibraltar where
  full_capacity : ℕ
  family_units : ℕ
  members_per_family : ℕ

/-- Calculates the starting population for a stable journey --/
def starting_population (ship : Gibraltar) : ℕ :=
  ship.full_capacity / 3 - 100

/-- Theorem: The starting population for a stable journey is 300 people --/
theorem stable_journey_population (ship : Gibraltar) 
  (h1 : ship.family_units = 300)
  (h2 : ship.members_per_family = 4)
  (h3 : ship.full_capacity = ship.family_units * ship.members_per_family) :
  starting_population ship = 300 := by
  sorry

#eval starting_population { full_capacity := 1200, family_units := 300, members_per_family := 4 }

end stable_journey_population_l3055_305504


namespace product_49_sum_14_l3055_305542

theorem product_49_sum_14 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 49 →
  a + b + c + d = 14 :=
by sorry

end product_49_sum_14_l3055_305542


namespace ad_greater_than_bc_l3055_305576

theorem ad_greater_than_bc (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
sorry

end ad_greater_than_bc_l3055_305576


namespace madeline_water_intake_l3055_305559

structure WaterBottle where
  capacity : ℕ

structure Activity where
  name : String
  goal : ℕ
  bottle : WaterBottle
  refills : ℕ

def total_intake (activities : List Activity) : ℕ :=
  activities.foldl (λ acc activity => acc + activity.bottle.capacity * (activity.refills + 1)) 0

def madeline_water_plan : List Activity :=
  [{ name := "Morning yoga", goal := 15, bottle := { capacity := 8 }, refills := 1 },
   { name := "Work", goal := 35, bottle := { capacity := 12 }, refills := 2 },
   { name := "Afternoon jog", goal := 20, bottle := { capacity := 16 }, refills := 1 },
   { name := "Evening leisure", goal := 30, bottle := { capacity := 8 }, refills := 1 },
   { name := "Evening leisure", goal := 30, bottle := { capacity := 16 }, refills := 1 }]

theorem madeline_water_intake :
  total_intake madeline_water_plan = 132 := by
  sorry

end madeline_water_intake_l3055_305559


namespace absolute_value_plus_reciprocal_zero_l3055_305543

theorem absolute_value_plus_reciprocal_zero (x : ℝ) :
  x ≠ 0 ∧ |x| + 1/x = 0 → x = -1 :=
by
  sorry

end absolute_value_plus_reciprocal_zero_l3055_305543


namespace chess_tournament_games_l3055_305519

theorem chess_tournament_games (n : ℕ) (h : n = 20) : 
  (n * (n - 1)) = 380 := by
  sorry

end chess_tournament_games_l3055_305519


namespace fourth_bell_interval_l3055_305546

theorem fourth_bell_interval 
  (bell1 bell2 bell3 : ℕ) 
  (h1 : bell1 = 5)
  (h2 : bell2 = 8)
  (h3 : bell3 = 11)
  (h4 : ∃ bell4 : ℕ, Nat.lcm (Nat.lcm (Nat.lcm bell1 bell2) bell3) bell4 = 1320) :
  ∃ bell4 : ℕ, bell4 = 1320 ∧ 
    Nat.lcm (Nat.lcm (Nat.lcm bell1 bell2) bell3) bell4 = 1320 :=
by sorry

end fourth_bell_interval_l3055_305546


namespace frog_corner_probability_l3055_305555

/-- Represents a position on the 3x3 grid -/
inductive Position
| Center
| Edge
| Corner

/-- Represents the state of the frog's movement -/
structure State where
  position : Position
  hops : Nat

/-- Transition function for the frog's movement -/
def transition (s : State) : List State := sorry

/-- Probability of reaching a corner after exactly 4 hops -/
def probability_corner_4_hops : ℚ := sorry

/-- Main theorem stating the probability of reaching a corner after exactly 4 hops -/
theorem frog_corner_probability :
  probability_corner_4_hops = 217 / 256 := by sorry

end frog_corner_probability_l3055_305555


namespace min_sum_positive_reals_l3055_305501

theorem min_sum_positive_reals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2 * a + 8 * b - a * b = 0 → x + y ≤ a + b ∧ x + y = 6 :=
sorry

end min_sum_positive_reals_l3055_305501


namespace cube_sum_theorem_l3055_305547

theorem cube_sum_theorem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : 
  x^3 + y^3 = 85/2 := by
sorry

end cube_sum_theorem_l3055_305547


namespace unique_k_solution_l3055_305561

/-- The function f(x) = x^2 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The theorem stating that k = 2 is the only solution -/
theorem unique_k_solution :
  ∃! k : ℝ, ∀ x : ℝ, f (x + k) = x^2 + 2*x + 1 ∧ k = 2 := by
  sorry

end unique_k_solution_l3055_305561


namespace boxtimes_self_not_always_zero_l3055_305523

-- Define the ⊠ operation
def boxtimes (x y : ℝ) : ℝ := |x + y|

-- Statement to be proven false
theorem boxtimes_self_not_always_zero :
  ¬ (∀ x : ℝ, boxtimes x x = 0) := by
sorry

end boxtimes_self_not_always_zero_l3055_305523


namespace trains_meeting_time_l3055_305515

/-- The time taken for two trains to meet under specific conditions -/
theorem trains_meeting_time : 
  let train1_length : ℝ := 300
  let train1_crossing_time : ℝ := 20
  let train2_length : ℝ := 450
  let train2_speed_kmh : ℝ := 90
  let train1_speed : ℝ := train1_length / train1_crossing_time
  let train2_speed : ℝ := train2_speed_kmh * 1000 / 3600
  let relative_speed : ℝ := train1_speed + train2_speed
  let total_distance : ℝ := train1_length + train2_length
  let meeting_time : ℝ := total_distance / relative_speed
  meeting_time = 18.75 := by sorry

end trains_meeting_time_l3055_305515


namespace competition_score_l3055_305506

theorem competition_score (correct_points incorrect_points total_questions final_score : ℕ) 
  (h1 : correct_points = 6)
  (h2 : incorrect_points = 3)
  (h3 : total_questions = 15)
  (h4 : final_score = 36) :
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_points = final_score ∧
    correct_answers = 9 := by
  sorry

#check competition_score

end competition_score_l3055_305506


namespace perpendicular_sum_difference_l3055_305513

/-- Given unit vectors a and b in the plane, prove that (a + b) is perpendicular to (a - b) -/
theorem perpendicular_sum_difference (a b : ℝ × ℝ) 
  (ha : a = (5/13, 12/13)) 
  (hb : b = (4/5, 3/5)) 
  (unit_a : a.1^2 + a.2^2 = 1) 
  (unit_b : b.1^2 + b.2^2 = 1) : 
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0 := by
sorry

end perpendicular_sum_difference_l3055_305513


namespace john_sublet_count_l3055_305540

/-- The number of people John sublets his apartment to -/
def num_subletters : ℕ := by sorry

/-- Monthly payment per subletter in dollars -/
def subletter_payment : ℕ := 400

/-- John's monthly rent in dollars -/
def john_rent : ℕ := 900

/-- John's annual profit in dollars -/
def annual_profit : ℕ := 3600

/-- Number of months in a year -/
def months_per_year : ℕ := 12

theorem john_sublet_count : 
  num_subletters * subletter_payment * months_per_year - john_rent * months_per_year = annual_profit → 
  num_subletters = 3 := by sorry

end john_sublet_count_l3055_305540


namespace range_of_a_l3055_305532

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

theorem range_of_a (a : ℝ) :
  (a ≠ 0) →
  (∀ x, ¬q x → ¬p x a) →
  (a ≥ 2 ∨ a ≤ -4) :=
by sorry

end range_of_a_l3055_305532


namespace sum_of_38_and_twice_43_l3055_305592

theorem sum_of_38_and_twice_43 : 38 + 2 * 43 = 124 := by
  sorry

end sum_of_38_and_twice_43_l3055_305592


namespace smallest_number_of_students_l3055_305514

/-- Represents the number of students in each grade --/
structure Students where
  ninth : ℕ
  eighth : ℕ
  seventh : ℕ

/-- The ratio of 9th-graders to 7th-graders is 4:5 --/
def ratio_ninth_to_seventh (s : Students) : Prop :=
  5 * s.ninth = 4 * s.seventh

/-- The ratio of 9th-graders to 8th-graders is 7:6 --/
def ratio_ninth_to_eighth (s : Students) : Prop :=
  6 * s.ninth = 7 * s.eighth

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.ninth + s.eighth + s.seventh

/-- The statement to be proved --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    ratio_ninth_to_seventh s ∧
    ratio_ninth_to_eighth s ∧
    total_students s = 87 ∧
    (∀ (t : Students),
      ratio_ninth_to_seventh t ∧
      ratio_ninth_to_eighth t →
      total_students t ≥ 87) := by
  sorry

end smallest_number_of_students_l3055_305514


namespace five_dozen_apple_cost_l3055_305545

/-- The cost of apples given the number of dozens and the price -/
def apple_cost (dozens : ℚ) (price : ℚ) : ℚ := dozens * (price / 4)

/-- Theorem: If 4 dozen apples cost $31.20, then 5 dozen apples at the same rate will cost $39.00 -/
theorem five_dozen_apple_cost :
  apple_cost 5 31.20 = 39.00 :=
sorry

end five_dozen_apple_cost_l3055_305545


namespace quadratic_roots_sum_product_l3055_305531

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ r₁ r₂ : ℝ, r₁ + r₂ = 4 ∧ r₁ * r₂ = 6 ∧ 
      3 * r₁^2 - p * r₁ + q = 0 ∧ 
      3 * r₂^2 - p * r₂ + q = 0)) → 
  p = 12 ∧ q = 18 := by
sorry

end quadratic_roots_sum_product_l3055_305531


namespace plain_lemonade_sales_l3055_305556

/-- The number of glasses of plain lemonade sold -/
def plain_lemonade_glasses : ℕ := 36

/-- The price of plain lemonade in dollars -/
def plain_lemonade_price : ℚ := 3/4

/-- The total revenue from strawberry lemonade in dollars -/
def strawberry_revenue : ℕ := 16

/-- The revenue difference between plain and strawberry lemonade in dollars -/
def revenue_difference : ℕ := 11

theorem plain_lemonade_sales :
  plain_lemonade_glasses * plain_lemonade_price = 
    (strawberry_revenue + revenue_difference : ℚ) := by sorry

end plain_lemonade_sales_l3055_305556


namespace hypotenuse_of_special_triangle_l3055_305562

-- Define a right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  angle_opposite_leg1 : ℝ
  is_right_triangle : leg1^2 + leg2^2 = hypotenuse^2

-- Theorem statement
theorem hypotenuse_of_special_triangle 
  (triangle : RightTriangle)
  (h1 : triangle.leg1 = 15)
  (h2 : triangle.angle_opposite_leg1 = 30 * π / 180) :
  triangle.hypotenuse = 30 :=
sorry

end hypotenuse_of_special_triangle_l3055_305562


namespace total_topping_combinations_l3055_305521

/-- Represents the number of cheese options -/
def cheese_options : ℕ := 3

/-- Represents the number of meat options -/
def meat_options : ℕ := 4

/-- Represents the number of vegetable options -/
def vegetable_options : ℕ := 5

/-- Represents whether pepperoni is a meat option -/
def pepperoni_is_meat_option : Prop := True

/-- Represents whether peppers is a vegetable option -/
def peppers_is_vegetable_option : Prop := True

/-- Represents the restriction that pepperoni and peppers cannot be chosen together -/
def pepperoni_peppers_restriction : Prop := True

/-- Theorem stating the total number of pizza topping combinations -/
theorem total_topping_combinations : 
  cheese_options * meat_options * vegetable_options - 
  cheese_options * (meat_options - 1) = 57 := by
  sorry


end total_topping_combinations_l3055_305521


namespace factorization_proof_l3055_305507

theorem factorization_proof (m n : ℝ) : 4 * m^2 * n - 4 * n^3 = 4 * n * (m + n) * (m - n) := by
  sorry

end factorization_proof_l3055_305507


namespace heracles_age_l3055_305596

/-- Proves that Heracles' current age is 10 years, given the conditions stated in the problem. -/
theorem heracles_age : ∃ (H : ℕ), 
  (∀ (A : ℕ), A = H + 7 → A + 3 = 2 * H) → H = 10 := by
  sorry

end heracles_age_l3055_305596


namespace ellipse_to_hyperbola_l3055_305518

/-- Given an ellipse with equation x²/4 + y²/2 = 1, 
    prove that the equation of the hyperbola with its vertices at the foci of the ellipse 
    and its foci at the vertices of the ellipse is x²/2 - y²/2 = 1 -/
theorem ellipse_to_hyperbola (x y : ℝ) :
  (x^2 / 4 + y^2 / 2 = 1) →
  ∃ (a b : ℝ), (a^2 = 2 ∧ b^2 = 2) ∧
  (x^2 / a^2 - y^2 / b^2 = 1) :=
sorry

end ellipse_to_hyperbola_l3055_305518


namespace composite_function_value_l3055_305590

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x + 3

-- State the theorem
theorem composite_function_value (c d : ℝ) :
  (∀ x, f c (g c x) = 15 * x + d) → d = 18 := by
  sorry

end composite_function_value_l3055_305590


namespace eighth_odd_multiple_of_five_l3055_305572

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

theorem eighth_odd_multiple_of_five : 
  ∀ (a d : ℕ),
    a = 5 → 
    d = 10 → 
    (∀ n : ℕ, n > 0 → arithmetic_sequence a d n % 2 = 1) →
    (∀ n : ℕ, n > 0 → arithmetic_sequence a d n % 5 = 0) →
    arithmetic_sequence a d 8 = 75 :=
by sorry

end eighth_odd_multiple_of_five_l3055_305572


namespace minimize_square_root_difference_l3055_305525

theorem minimize_square_root_difference (p : ℕ) (h_p : Nat.Prime p) (h_p_odd : Odd p) :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ x ≤ y ∧
    (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 0) ∧
    (∀ (a b : ℕ), a > 0 → b > 0 → a ≤ b → 
      Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0 →
      Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≤ Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) ∧
    x = (p - 1) / 2 ∧ y = (p + 1) / 2 := by
  sorry

end minimize_square_root_difference_l3055_305525


namespace not_integer_proofs_l3055_305591

theorem not_integer_proofs (a b c d : ℝ) (n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  let E := (a/(a+b+d)) + (b/(b+c+a)) + (c/(c+d+b)) + (d/(d+a+c))
  (1 < E ∧ E < 2) ∧ (n < Real.sqrt (n^2 + n) ∧ Real.sqrt (n^2 + n) < n + 1) := by
  sorry

end not_integer_proofs_l3055_305591


namespace average_of_remaining_numbers_l3055_305580

theorem average_of_remaining_numbers
  (total : ℝ)
  (group1 : ℝ)
  (group2 : ℝ)
  (h1 : total = 6 * 6.40)
  (h2 : group1 = 2 * 6.2)
  (h3 : group2 = 2 * 6.1) :
  (total - group1 - group2) / 2 = 6.9 := by
  sorry

end average_of_remaining_numbers_l3055_305580


namespace least_number_satisfying_conditions_l3055_305558

def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

def leaves_remainder_2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_11 n ∧ 
  (∀ d : ℕ, 3 ≤ d → d ≤ 7 → leaves_remainder_2 n d)

theorem least_number_satisfying_conditions : 
  satisfies_conditions 3782 ∧ 
  ∀ m : ℕ, m < 3782 → ¬(satisfies_conditions m) :=
sorry

end least_number_satisfying_conditions_l3055_305558


namespace roses_cut_is_difference_l3055_305541

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that the number of roses Mary cut is the difference between the final and initial number of roses -/
theorem roses_cut_is_difference (initial_roses final_roses : ℕ) 
  (h : final_roses ≥ initial_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

#eval roses_cut 6 16  -- Should evaluate to 10

end roses_cut_is_difference_l3055_305541


namespace largest_rectangle_area_l3055_305565

/-- Represents a rectangular area within a square grid -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a square grid -/
structure Grid where
  size : Nat
  center : Nat × Nat

/-- Checks if a rectangle contains the center of a grid -/
def containsCenter (r : Rectangle) (g : Grid) : Prop :=
  ∃ (x y : Nat), x ≥ 1 ∧ x ≤ r.width ∧ y ≥ 1 ∧ y ≤ r.height ∧ 
    (x + (g.size - r.width) / 2, y + (g.size - r.height) / 2) = g.center

/-- Checks if a rectangle fits within a grid -/
def fitsInGrid (r : Rectangle) (g : Grid) : Prop :=
  r.width ≤ g.size ∧ r.height ≤ g.size

/-- The area of a rectangle -/
def area (r : Rectangle) : Nat :=
  r.width * r.height

/-- The theorem to be proved -/
theorem largest_rectangle_area (g : Grid) (r : Rectangle) : 
  g.size = 11 → 
  g.center = (6, 6) → 
  fitsInGrid r g → 
  ¬containsCenter r g → 
  area r ≤ 55 := by
  sorry

end largest_rectangle_area_l3055_305565


namespace class_average_problem_l3055_305550

theorem class_average_problem (class_size : ℝ) (h_positive : class_size > 0) :
  let group1_size := 0.2 * class_size
  let group2_size := 0.5 * class_size
  let group3_size := class_size - group1_size - group2_size
  let group1_avg := 80
  let group2_avg := 60
  let overall_avg := 58
  ∃ (group3_avg : ℝ),
    (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg) / class_size = overall_avg ∧
    group3_avg = 40 := by
sorry

end class_average_problem_l3055_305550


namespace polynomial_equality_implies_sum_l3055_305595

theorem polynomial_equality_implies_sum (b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃) * (x^2 + b₄*x + c₄)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ + b₄*c₄ = -1 := by
sorry

end polynomial_equality_implies_sum_l3055_305595


namespace one_positive_integer_solution_l3055_305503

theorem one_positive_integer_solution : 
  ∃! (x : ℕ), x > 0 ∧ 24 - 6 * x > 12 :=
by sorry

end one_positive_integer_solution_l3055_305503


namespace polynomial_divisibility_l3055_305574

theorem polynomial_divisibility (n : ℕ) (h : n > 1) :
  ∃ Q : Polynomial ℂ, x^(4*n+3) + x^(4*n+1) + x^(4*n-2) + x^8 = (x^2 + 1) * Q := by
  sorry

#check polynomial_divisibility

end polynomial_divisibility_l3055_305574


namespace range_of_a_l3055_305566

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Icc 1 4 ∪ Set.Iic (-2) :=
sorry

end range_of_a_l3055_305566


namespace function_expression_l3055_305571

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 2) = 2 * x + 3) :
  ∀ x, f x = 2 * x - 1 := by
  sorry

end function_expression_l3055_305571


namespace min_board_size_is_77_l3055_305588

/-- A domino placement on a square board. -/
structure DominoPlacement where
  n : ℕ  -- Size of the square board
  dominoes : ℕ  -- Number of dominoes placed

/-- Checks if the domino placement is valid. -/
def is_valid_placement (p : DominoPlacement) : Prop :=
  p.dominoes * 2 = 2008 ∧  -- Total area covered by dominoes
  (p.n + 1)^2 ≥ p.dominoes * 6  -- Extended board can fit dominoes with shadows

/-- The minimum board size for a valid domino placement. -/
def min_board_size : ℕ := 77

/-- Theorem stating that 77 is the minimum board size for a valid domino placement. -/
theorem min_board_size_is_77 :
  ∀ p : DominoPlacement, is_valid_placement p → p.n ≥ min_board_size :=
by sorry

end min_board_size_is_77_l3055_305588


namespace sum_of_imaginary_parts_l3055_305584

/-- Given three complex numbers with specific conditions, prove that s+u = 1 -/
theorem sum_of_imaginary_parts (p q r s t u : ℝ) : 
  q = 5 → 
  p = -r - 2*t → 
  Complex.mk (p + r + t) (q + s + u) = Complex.I * 6 → 
  s + u = 1 := by
sorry

end sum_of_imaginary_parts_l3055_305584


namespace original_number_proof_l3055_305579

theorem original_number_proof (x y : ℝ) : 
  10 * x + 22 * y = 780 → 
  y = 34 → 
  x + y = 37.2 :=
by sorry

end original_number_proof_l3055_305579


namespace solve_for_y_l3055_305511

/-- Custom operation € defined for real numbers -/
def custom_op (x y : ℝ) : ℝ := 2 * x * y

/-- Theorem stating that under given conditions, y must equal 5 -/
theorem solve_for_y (y : ℝ) :
  (custom_op 7 (custom_op 4 y) = 560) → y = 5 := by
  sorry

end solve_for_y_l3055_305511


namespace missing_score_is_90_l3055_305553

def known_scores : List ℕ := [85, 90, 87, 93]

theorem missing_score_is_90 (x : ℕ) :
  (x :: known_scores).sum / (x :: known_scores).length = 89 →
  x = 90 := by
  sorry

end missing_score_is_90_l3055_305553


namespace simplify_trig_expression_l3055_305573

theorem simplify_trig_expression :
  1 / Real.sqrt (1 + Real.tan (160 * π / 180) ^ 2) = -Real.cos (160 * π / 180) := by
  sorry

end simplify_trig_expression_l3055_305573


namespace regular_polygon_sides_l3055_305554

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end regular_polygon_sides_l3055_305554


namespace prime_factors_equation_l3055_305587

theorem prime_factors_equation (x : ℕ) : 22 + x + 2 = 29 → x = 5 := by
  sorry

end prime_factors_equation_l3055_305587


namespace mathborough_rainfall_2005_l3055_305582

/-- Rainfall data for Mathborough from 2003 to 2005 -/
structure RainfallData where
  rainfall_2003 : ℝ
  increase_2004 : ℝ
  increase_2005 : ℝ

/-- Calculate the total rainfall in Mathborough for 2005 -/
def totalRainfall2005 (data : RainfallData) : ℝ :=
  12 * (data.rainfall_2003 + data.increase_2004 + data.increase_2005)

/-- Theorem stating the total rainfall in Mathborough for 2005 -/
theorem mathborough_rainfall_2005 (data : RainfallData)
  (h1 : data.rainfall_2003 = 50)
  (h2 : data.increase_2004 = 5)
  (h3 : data.increase_2005 = 3) :
  totalRainfall2005 data = 696 := by
  sorry

#eval totalRainfall2005 ⟨50, 5, 3⟩

end mathborough_rainfall_2005_l3055_305582


namespace parking_tickets_l3055_305564

theorem parking_tickets (total : ℕ) (alan : ℕ) (marcy : ℕ) 
  (h1 : total = 150)
  (h2 : alan = 26)
  (h3 : marcy = 5 * alan)
  (h4 : total = alan + marcy) :
  total - marcy = 104 := by
  sorry

end parking_tickets_l3055_305564


namespace intersection_of_sets_l3055_305528

theorem intersection_of_sets : 
  let A : Set ℤ := {1, 2, 3}
  let B : Set ℤ := {-2, 2}
  A ∩ B = {2} := by sorry

end intersection_of_sets_l3055_305528


namespace solve_system_l3055_305583

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  p = 52 / 11 := by
sorry

end solve_system_l3055_305583


namespace sine_shift_overlap_l3055_305578

/-- The smallest positive value of ω that makes the sine function overlap with its shifted version -/
theorem sine_shift_overlap : ∃ ω : ℝ, ω > 0 ∧ 
  (∀ x : ℝ, Real.sin (ω * x + π / 3) = Real.sin (ω * (x - π / 3) + π / 3)) ∧
  (∀ ω' : ℝ, ω' > 0 → 
    (∀ x : ℝ, Real.sin (ω' * x + π / 3) = Real.sin (ω' * (x - π / 3) + π / 3)) → 
    ω ≤ ω') ∧
  ω = 2 * π := by
sorry

end sine_shift_overlap_l3055_305578


namespace right_triangle_area_l3055_305567

/-- The area of a right-angled triangle with base 12 cm and height 15 cm is 90 square centimeters -/
theorem right_triangle_area : 
  ∀ (base height area : ℝ), 
  base = 12 → 
  height = 15 → 
  area = (1/2) * base * height → 
  area = 90 := by
sorry

end right_triangle_area_l3055_305567


namespace max_value_sqrt_sum_l3055_305529

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  Real.sqrt (a + 1) + Real.sqrt (b + 2) ≤ 2 * Real.sqrt 3 := by
  sorry

end max_value_sqrt_sum_l3055_305529


namespace certain_number_solution_l3055_305568

theorem certain_number_solution : 
  ∃ x : ℝ, (3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5) = 800.0000000000001 ∧ x = 2.5 := by
  sorry

end certain_number_solution_l3055_305568


namespace vector_linear_combination_l3055_305530

/-- Given vectors a, b, and c in R^2, prove that if c = x*a + y*b,
    then x + y = 8/3 -/
theorem vector_linear_combination (a b c : ℝ × ℝ) (x y : ℝ) 
    (h1 : a = (2, 3))
    (h2 : b = (3, 3))
    (h3 : c = (7, 8))
    (h4 : c = x • a + y • b) :
  x + y = 8/3 := by
  sorry

end vector_linear_combination_l3055_305530


namespace optimal_strategy_l3055_305549

/-- Represents the price of bananas on each day of Marina's trip. -/
def banana_prices : List ℝ := [1, 5, 1, 6, 7, 8, 1, 8, 7, 2, 7, 8, 1, 9, 2, 8, 7, 1]

/-- Represents the optimal buying strategy for bananas. -/
def buying_strategy : List ℕ := [1, 1, 1, 4, 0, 0, 1, 0, 1, 4, 1, 0, 0, 0, 3, 0, 0, 2, 0]

/-- The number of days in Marina's trip. -/
def trip_length : ℕ := 18

/-- The maximum number of days a banana can be eaten after purchase. -/
def max_banana_freshness : ℕ := 4

/-- Calculates the total cost of bananas based on a given buying strategy. -/
def total_cost (strategy : List ℕ) : ℝ :=
  List.sum (List.zipWith (· * ·) strategy banana_prices)

/-- Checks if a given buying strategy is valid according to the problem constraints. -/
def is_valid_strategy (strategy : List ℕ) : Prop :=
  strategy.length = trip_length + 1 ∧
  List.sum strategy = trip_length ∧
  ∀ i, i < trip_length → List.sum (List.take (min max_banana_freshness (trip_length - i)) (List.drop i strategy)) ≥ 1

/-- Theorem stating that the given buying strategy is optimal. -/
theorem optimal_strategy :
  is_valid_strategy buying_strategy ∧
  ∀ other_strategy, is_valid_strategy other_strategy →
    total_cost buying_strategy ≤ total_cost other_strategy :=
sorry

end optimal_strategy_l3055_305549


namespace cement_warehouse_distribution_l3055_305520

theorem cement_warehouse_distribution (total : ℕ) (extra : ℕ) (multiplier : ℕ) 
  (warehouseA : ℕ) (warehouseB : ℕ) : 
  total = 462 → 
  extra = 32 → 
  multiplier = 4 →
  total = warehouseA + warehouseB → 
  warehouseA = multiplier * warehouseB + extra →
  warehouseA = 376 ∧ warehouseB = 86 := by
  sorry

end cement_warehouse_distribution_l3055_305520


namespace quadratic_real_solution_l3055_305581

theorem quadratic_real_solution (m : ℝ) : 
  (∃ z : ℝ, z^2 + Complex.I * z + m = 0) ↔ m = 0 := by
  sorry

end quadratic_real_solution_l3055_305581


namespace river_speed_l3055_305539

/-- Proves that the speed of the river is 1.2 kmph given the conditions of the rowing problem -/
theorem river_speed (still_water_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) :
  still_water_speed = 10 →
  total_time = 1 →
  total_distance = 9.856 →
  ∃ (river_speed : ℝ),
    river_speed = 1.2 ∧
    total_distance = (still_water_speed - river_speed) * (total_time / 2) +
                     (still_water_speed + river_speed) * (total_time / 2) :=
by
  sorry

end river_speed_l3055_305539


namespace quadratic_inequality_properties_l3055_305527

theorem quadratic_inequality_properties (a b c : ℝ) :
  (∀ x, (a * x^2 + b * x + c ≤ 0) ↔ (x ≤ -2 ∨ x ≥ 3)) →
  (a < 0 ∧
   (∀ x, (a * x + c > 0) ↔ x < 6) ∧
   (∀ x, (c * x^2 + b * x + a < 0) ↔ (-1/2 < x ∧ x < 1/3))) :=
by sorry

end quadratic_inequality_properties_l3055_305527


namespace stating_least_possible_area_l3055_305536

/-- Represents the length of a side of a square in centimeters. -/
def SideLength : ℝ := 5

/-- The lower bound of the actual side length when measured to the nearest centimeter. -/
def LowerBound : ℝ := SideLength - 0.5

/-- Calculates the area of a square given its side length. -/
def SquareArea (side : ℝ) : ℝ := side * side

/-- 
Theorem stating that the least possible area of a square with sides measured as 5 cm 
to the nearest centimeter is 20.25 cm².
-/
theorem least_possible_area :
  SquareArea LowerBound = 20.25 := by sorry

end stating_least_possible_area_l3055_305536


namespace initial_speed_calculation_l3055_305502

theorem initial_speed_calculation (distance : ℝ) (fast_speed : ℝ) (time_diff : ℝ) 
  (h1 : distance = 24)
  (h2 : fast_speed = 12)
  (h3 : time_diff = 2/3) : 
  ∃ v : ℝ, v > 0 ∧ distance / v - distance / fast_speed = time_diff ∧ v = 9 := by
sorry

end initial_speed_calculation_l3055_305502


namespace smallest_rational_number_l3055_305544

theorem smallest_rational_number : ∀ (a b c d : ℚ), 
  a = 0 → b = -1/2 → c = -1/3 → d = 4 →
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by
  sorry

end smallest_rational_number_l3055_305544


namespace canned_food_bins_l3055_305586

theorem canned_food_bins (soup : Real) (vegetables : Real) (pasta : Real)
  (h1 : soup = 0.12)
  (h2 : vegetables = 0.12)
  (h3 : pasta = 0.5) :
  soup + vegetables + pasta = 0.74 := by
  sorry

end canned_food_bins_l3055_305586


namespace expression_evaluation_l3055_305512

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 2
  ((x + 2*y)^2 + (3*x + y)*(3*x - y) - 3*y*(y - x)) / (2*x) = -3 := by
  sorry

end expression_evaluation_l3055_305512


namespace solve_exponential_equation_l3055_305575

theorem solve_exponential_equation : 
  ∃ x : ℝ, (125 : ℝ)^4 = 5^x ∧ x = 12 := by
  sorry

end solve_exponential_equation_l3055_305575


namespace sphere_expansion_l3055_305534

/-- Given a sphere with initial radius 1 and final radius m, 
    if the volume expansion rate is 28π/3, then m = 2 -/
theorem sphere_expansion (m : ℝ) : 
  m > 0 →  -- Ensure m is positive (as it's a radius)
  (4 * π / 3 * (m^3 - 1)) / (m - 1) = 28 * π / 3 →
  m = 2 :=
by
  sorry


end sphere_expansion_l3055_305534


namespace infinite_nested_sqrt_three_l3055_305517

theorem infinite_nested_sqrt_three : ∃ x > 0, x^2 = 3 + 2*x ∧ x = 3 := by
  sorry

end infinite_nested_sqrt_three_l3055_305517


namespace power_addition_equality_l3055_305522

theorem power_addition_equality : 2^345 + 9^4 / 9^2 = 2^345 + 81 := by
  sorry

end power_addition_equality_l3055_305522


namespace min_photos_theorem_l3055_305551

/-- Represents the number of photographs for each grade --/
structure PhotoDistribution where
  total : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ
  seventh : ℕ
  first_to_third : ℕ

/-- The minimum number of photographs needed to ensure at least 15 from one grade (4th to 7th) --/
def min_photos_for_fifteen (d : PhotoDistribution) : ℕ := 
  d.first_to_third + 4 * 14 + 1

/-- The theorem stating the minimum number of photographs needed --/
theorem min_photos_theorem (d : PhotoDistribution) 
  (h_total : d.total = 130)
  (h_fourth : d.fourth = 35)
  (h_fifth : d.fifth = 30)
  (h_sixth : d.sixth = 25)
  (h_seventh : d.seventh = 20)
  (h_first_to_third : d.first_to_third = d.total - (d.fourth + d.fifth + d.sixth + d.seventh)) :
  min_photos_for_fifteen d = 77 := by
  sorry

#eval min_photos_for_fifteen ⟨130, 35, 30, 25, 20, 20⟩

end min_photos_theorem_l3055_305551


namespace circle_radii_sum_l3055_305597

theorem circle_radii_sum (r₁ r₂ : ℝ) : 
  (∃ (a : ℝ), (2 - a)^2 + (5 - a)^2 = a^2 ∧ 
               r₁ = a ∧ 
               (∃ (b : ℝ), b^2 - 14*b + 29 = 0 ∧ r₂ = b)) →
  r₁ + r₂ = 14 := by
sorry


end circle_radii_sum_l3055_305597


namespace bullseye_value_l3055_305598

/-- 
Given a dart game with the following conditions:
- Three darts are thrown
- One dart is a bullseye worth B points
- One dart completely misses (0 points)
- One dart is worth half the bullseye points
- The total score is 75 points

Prove that the bullseye is worth 50 points
-/
theorem bullseye_value (B : ℝ) 
  (total_score : B + 0 + B/2 = 75) : 
  B = 50 := by
  sorry

end bullseye_value_l3055_305598


namespace swimming_pool_count_l3055_305589

theorem swimming_pool_count (total : ℕ) (garage : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 70 → garage = 50 → both = 35 → neither = 15 → 
  ∃ pool : ℕ, pool = 40 ∧ total = garage + pool - both + neither :=
by sorry

end swimming_pool_count_l3055_305589


namespace wren_population_decline_l3055_305516

theorem wren_population_decline (n : ℕ) : (∀ k : ℕ, k < n → (0.7 : ℝ) ^ k ≥ 0.1) ∧ (0.7 : ℝ) ^ n < 0.1 → n = 7 := by
  sorry

end wren_population_decline_l3055_305516


namespace english_only_enrollment_l3055_305538

/-- Represents the enrollment data for a class with English and German courses -/
structure ClassEnrollment where
  total : Nat
  both : Nat
  german : Nat

/-- Calculates the number of students enrolled only in English -/
def studentsOnlyEnglish (c : ClassEnrollment) : Nat :=
  c.total - c.german

/-- Theorem stating that 28 students are enrolled only in English -/
theorem english_only_enrollment (c : ClassEnrollment) 
  (h1 : c.total = 50)
  (h2 : c.both = 12)
  (h3 : c.german = 22)
  (h4 : c.total = studentsOnlyEnglish c + c.german) :
  studentsOnlyEnglish c = 28 := by
  sorry

#eval studentsOnlyEnglish { total := 50, both := 12, german := 22 }

end english_only_enrollment_l3055_305538


namespace valid_y_characterization_l3055_305509

/-- The set of y values in [0, 2π] for which sin(x+y) ≥ sin(x) - sin(y) holds for all x in [0, 2π] -/
def valid_y_set : Set ℝ :=
  {y | 0 ≤ y ∧ y ≤ 2 * Real.pi ∧ 
    ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi → Real.sin (x + y) ≥ Real.sin x - Real.sin y}

theorem valid_y_characterization :
  valid_y_set = {0, 2 * Real.pi} := by sorry

end valid_y_characterization_l3055_305509


namespace meal_combinations_l3055_305563

def fruit_count : ℕ := 3
def salad_count : ℕ := 4
def dessert_count : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem meal_combinations :
  fruit_count * choose salad_count 2 * dessert_count = 90 := by
  sorry

end meal_combinations_l3055_305563


namespace smallest_multiplier_for_perfect_cube_l3055_305593

def y : Nat := 2^3 * 3^5 * 4^5 * 5^4 * 6^3 * 7^5 * 8^2

def is_perfect_cube (n : Nat) : Prop :=
  ∃ m : Nat, n = m^3

theorem smallest_multiplier_for_perfect_cube :
  (∀ z < 350, ¬ is_perfect_cube (y * z)) ∧ is_perfect_cube (y * 350) :=
sorry

end smallest_multiplier_for_perfect_cube_l3055_305593


namespace chocolate_difference_l3055_305537

theorem chocolate_difference (t : ℚ) : 
  let sarah := (1 : ℚ) / 3 * t
  let andrew := (3 : ℚ) / 8 * t
  let cecily := t - (sarah + andrew)
  sarah - cecily = (1 : ℚ) / 24 * t := by sorry

end chocolate_difference_l3055_305537


namespace quadratic_increasing_l3055_305599

/-- Given a quadratic function y = (x - 1)^2 + 2, prove that y is increasing when x > 1 -/
theorem quadratic_increasing (x : ℝ) : 
  let y : ℝ → ℝ := λ x ↦ (x - 1)^2 + 2
  x > 1 → ∀ h > 0, y (x + h) > y x :=
by sorry

end quadratic_increasing_l3055_305599


namespace amber_amethyst_ratio_l3055_305535

/-- Given a necklace with 40 beads, 7 amethyst beads, and 19 turquoise beads,
    prove that the ratio of amber beads to amethyst beads is 2:1. -/
theorem amber_amethyst_ratio (total : ℕ) (amethyst : ℕ) (turquoise : ℕ) 
  (h1 : total = 40)
  (h2 : amethyst = 7)
  (h3 : turquoise = 19) :
  (total - amethyst - turquoise) / amethyst = 2 := by
  sorry

end amber_amethyst_ratio_l3055_305535


namespace polynomial_division_theorem_l3055_305548

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 - 8 = (x - 2) * (x^5 + 2*x^4 + 4*x^3 + 8*x^2 + 16*x + 32) + 56 := by
  sorry

end polynomial_division_theorem_l3055_305548


namespace integer_root_values_l3055_305569

/-- The polynomial for which we're finding integer roots -/
def P (a : ℤ) (x : ℤ) : ℤ := x^3 + 2*x^2 + a*x + 10

/-- The set of possible values for a -/
def A : Set ℤ := {-1210, -185, -26, -13, -11, -10, 65, 790}

/-- Theorem stating that A contains exactly the values of a for which P has an integer root -/
theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, P a x = 0) ↔ a ∈ A :=
sorry

end integer_root_values_l3055_305569


namespace range_of_a_l3055_305533

def has_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 4 = 0

def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x ≥ 3, (4*x + a) ≥ 0

theorem range_of_a (a : ℝ) :
  has_real_roots a ∧ is_increasing_on_interval a →
  a ∈ Set.Icc (-12) (-4) ∪ Set.Ioi 4 :=
sorry

end range_of_a_l3055_305533


namespace factorization_x_squared_minus_xy_l3055_305526

theorem factorization_x_squared_minus_xy (x y : ℝ) : x^2 - x*y = x*(x - y) := by
  sorry

end factorization_x_squared_minus_xy_l3055_305526


namespace book_sale_price_l3055_305500

theorem book_sale_price (total_books : ℕ) (sold_books : ℕ) (unsold_books : ℕ) (total_amount : ℕ) : 
  sold_books = (2 : ℕ) * total_books / 3 →
  unsold_books = 36 →
  sold_books + unsold_books = total_books →
  total_amount = 288 →
  total_amount / sold_books = 4 := by
sorry

end book_sale_price_l3055_305500


namespace smallest_nonprime_with_large_factors_l3055_305557

def is_nonprime (n : ℕ) : Prop := ¬(Nat.Prime n) ∧ n > 1

def has_no_small_prime_factor (n : ℕ) : Prop := ∀ p, Nat.Prime p → p < 20 → ¬(p ∣ n)

theorem smallest_nonprime_with_large_factors : 
  ∃ n : ℕ, is_nonprime n ∧ has_no_small_prime_factor n ∧ 
  (∀ m : ℕ, m < n → ¬(is_nonprime m ∧ has_no_small_prime_factor m)) ∧
  n = 529 :=
sorry

end smallest_nonprime_with_large_factors_l3055_305557


namespace average_of_c_and_d_l3055_305570

theorem average_of_c_and_d (c d e : ℝ) : 
  (4 + 6 + 9 + c + d + e) / 6 = 20 → 
  e = c + 6 → 
  (c + d) / 2 = 47.5 := by
sorry

end average_of_c_and_d_l3055_305570


namespace pentagon_area_is_8_5_l3055_305552

-- Define the pentagon vertices
def pentagon_vertices : List (ℤ × ℤ) := [(0, 0), (1, 2), (3, 3), (4, 1), (2, 0)]

-- Define the function to calculate the area of the pentagon
def pentagon_area (vertices : List (ℤ × ℤ)) : ℚ :=
  sorry

-- Theorem statement
theorem pentagon_area_is_8_5 :
  pentagon_area pentagon_vertices = 17/2 := by sorry

end pentagon_area_is_8_5_l3055_305552


namespace simplify_expression_l3055_305524

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^2)^11 = 5368709120 := by
  sorry

end simplify_expression_l3055_305524
