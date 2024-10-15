import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_geometric_means_sum_l1840_184081

/-- Given real numbers a, b, c in geometric progression and non-zero real numbers x, y
    that are arithmetic means of a, b and b, c respectively, prove that a/x + b/y = 2 -/
theorem arithmetic_geometric_means_sum (a b c x y : ℝ) 
  (hgp : b^2 = a*c)  -- geometric progression condition
  (hx : x ≠ 0)       -- x is non-zero
  (hy : y ≠ 0)       -- y is non-zero
  (hax : 2*x = a + b)  -- x is arithmetic mean of a and b
  (hby : 2*y = b + c)  -- y is arithmetic mean of b and c
  : a/x + b/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_means_sum_l1840_184081


namespace NUMINAMATH_CALUDE_girls_from_maple_grove_l1840_184027

/-- Represents the number of students in different categories -/
structure StudentCounts where
  total : Nat
  girls : Nat
  boys : Nat
  pinecrest : Nat
  mapleGrove : Nat
  boysPinecrest : Nat

/-- The theorem stating that 40 girls are from Maple Grove School -/
theorem girls_from_maple_grove (s : StudentCounts)
  (h_total : s.total = 150)
  (h_girls : s.girls = 90)
  (h_boys : s.boys = 60)
  (h_pinecrest : s.pinecrest = 80)
  (h_mapleGrove : s.mapleGrove = 70)
  (h_boysPinecrest : s.boysPinecrest = 30)
  (h_total_sum : s.total = s.girls + s.boys)
  (h_school_sum : s.total = s.pinecrest + s.mapleGrove)
  : s.girls - (s.pinecrest - s.boysPinecrest) = 40 := by
  sorry


end NUMINAMATH_CALUDE_girls_from_maple_grove_l1840_184027


namespace NUMINAMATH_CALUDE_national_park_pines_l1840_184063

theorem national_park_pines (pines redwoods : ℕ) : 
  redwoods = pines + pines / 5 →
  pines + redwoods = 1320 →
  pines = 600 := by
sorry

end NUMINAMATH_CALUDE_national_park_pines_l1840_184063


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1840_184058

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 = 2 →                        -- given a₂ = 2
  a 5 = 1/4 →                      -- given a₅ = 1/4
  q = 1/2 :=                       -- prove q = 1/2
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1840_184058


namespace NUMINAMATH_CALUDE_square_area_error_l1840_184031

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.04)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 8.16 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l1840_184031


namespace NUMINAMATH_CALUDE_initial_milk_collected_l1840_184035

/-- Proves that the initial amount of milk collected equals 30,000 gallons -/
theorem initial_milk_collected (
  pumping_hours : ℕ)
  (pumping_rate : ℕ)
  (adding_hours : ℕ)
  (adding_rate : ℕ)
  (milk_left : ℕ)
  (h1 : pumping_hours = 4)
  (h2 : pumping_rate = 2880)
  (h3 : adding_hours = 7)
  (h4 : adding_rate = 1500)
  (h5 : milk_left = 28980)
  (h6 : ∃ initial_milk : ℕ, 
    initial_milk + adding_hours * adding_rate - pumping_hours * pumping_rate = milk_left) :
  ∃ initial_milk : ℕ, initial_milk = 30000 := by
sorry


end NUMINAMATH_CALUDE_initial_milk_collected_l1840_184035


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1840_184060

theorem quadratic_roots_to_coefficients 
  (a b p q : ℝ) 
  (h1 : Complex.I ^ 2 = -1) 
  (h2 : (2 + a * Complex.I) ^ 2 + p * (2 + a * Complex.I) + q = 0) 
  (h3 : (b + Complex.I) ^ 2 + p * (b + Complex.I) + q = 0) : 
  p = -4 ∧ q = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1840_184060


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1840_184061

theorem isosceles_triangle_base_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  (α = β ∨ α = γ ∨ β = γ) →  -- The triangle is isosceles
  (α = 110 ∨ β = 110 ∨ γ = 110) →  -- One of the angles is 110°
  (α = 35 ∨ β = 35 ∨ γ = 35) :=  -- One of the base angles is 35°
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1840_184061


namespace NUMINAMATH_CALUDE_equilateral_triangle_most_stable_l1840_184025

-- Define the shapes
inductive Shape
| EquilateralTriangle
| Square
| Parallelogram
| Trapezoid

-- Define stability as a function of shape properties
def stability (s : Shape) : ℝ :=
  match s with
  | Shape.EquilateralTriangle => 1
  | Shape.Square => 0.9
  | Shape.Parallelogram => 0.7
  | Shape.Trapezoid => 0.5

-- Define a predicate for being the most stable
def is_most_stable (s : Shape) : Prop :=
  ∀ t : Shape, stability s ≥ stability t

-- Theorem statement
theorem equilateral_triangle_most_stable :
  is_most_stable Shape.EquilateralTriangle :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_most_stable_l1840_184025


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l1840_184004

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -7 * x^2 + 4 * x - 6 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l1840_184004


namespace NUMINAMATH_CALUDE_gcd_problem_l1840_184065

theorem gcd_problem (x : ℤ) (h : ∃ k : ℤ, x = 17248 * k) :
  Int.gcd ((5*x+4)*(8*x+1)*(11*x+6)*(3*x+9)) x = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1840_184065


namespace NUMINAMATH_CALUDE_irrational_sqrt_3_rational_others_l1840_184088

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

-- Theorem statement
theorem irrational_sqrt_3_rational_others :
  IsIrrational (Real.sqrt 3) ∧
  IsRational 0 ∧
  IsRational (-2) ∧
  IsRational (1/2) := by
  sorry

end NUMINAMATH_CALUDE_irrational_sqrt_3_rational_others_l1840_184088


namespace NUMINAMATH_CALUDE_equation_solution_l1840_184056

theorem equation_solution (x : ℝ) : 
  x ≠ 3 → (-x^2 = (3*x - 3) / (x - 3)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1840_184056


namespace NUMINAMATH_CALUDE_college_student_count_l1840_184080

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The total number of students in the college -/
def College.total (c : College) : ℕ := c.boys + c.girls

/-- A college with a 5:7 ratio of boys to girls and 140 girls -/
def myCollege : College where
  girls := 140
  boys := 140 * 5 / 7

theorem college_student_count : myCollege.total = 240 := by
  sorry

end NUMINAMATH_CALUDE_college_student_count_l1840_184080


namespace NUMINAMATH_CALUDE_sports_club_members_l1840_184053

theorem sports_club_members (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 30 →
  badminton = 18 →
  tennis = 19 →
  both = 9 →
  total - (badminton + tennis - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l1840_184053


namespace NUMINAMATH_CALUDE_book_page_ratio_l1840_184009

theorem book_page_ratio (total_pages : ℕ) (intro_pages : ℕ) (text_pages : ℕ) 
  (h1 : total_pages = 98)
  (h2 : intro_pages = 11)
  (h3 : text_pages = 19)
  (h4 : text_pages = (total_pages - intro_pages - text_pages * 2) / 2) :
  (total_pages - intro_pages - text_pages * 2) / total_pages = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_page_ratio_l1840_184009


namespace NUMINAMATH_CALUDE_total_trips_is_seven_l1840_184003

/-- Calculates the number of trips needed to carry a given number of trays -/
def trips_needed (trays_per_trip : ℕ) (num_trays : ℕ) : ℕ :=
  (num_trays + trays_per_trip - 1) / trays_per_trip

/-- Proves that the total number of trips needed is 7 -/
theorem total_trips_is_seven (trays_per_trip : ℕ) (table1_trays : ℕ) (table2_trays : ℕ)
    (h1 : trays_per_trip = 3)
    (h2 : table1_trays = 15)
    (h3 : table2_trays = 5) :
    trips_needed trays_per_trip table1_trays + trips_needed trays_per_trip table2_trays = 7 := by
  sorry

#eval trips_needed 3 15 + trips_needed 3 5

end NUMINAMATH_CALUDE_total_trips_is_seven_l1840_184003


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1840_184093

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = (1/2) * x - 3/2) →  -- L1 in slope-intercept form
  (L2 P.1 P.2) →  -- L2 passes through P
  (∀ m1 m2, m1 = 1/2 ∧ m2 = -2 → m1 * m2 = -1) →  -- Perpendicular slopes multiply to -1
  ∃ m b, ∀ x y, L2 x y ↔ y = m * x + b ∧ m * (1/2) = -1  -- L2 is perpendicular to L1
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1840_184093


namespace NUMINAMATH_CALUDE_clares_money_l1840_184097

/-- The amount of money Clare's mother gave her --/
def money_from_mother : ℕ := sorry

/-- The number of loaves of bread Clare bought --/
def bread_count : ℕ := 4

/-- The number of cartons of milk Clare bought --/
def milk_count : ℕ := 2

/-- The cost of one loaf of bread in dollars --/
def bread_cost : ℕ := 2

/-- The cost of one carton of milk in dollars --/
def milk_cost : ℕ := 2

/-- The amount of money Clare has left after shopping --/
def money_left : ℕ := 35

/-- Theorem stating that the amount of money Clare's mother gave her is $47 --/
theorem clares_money : money_from_mother = 47 := by
  sorry

end NUMINAMATH_CALUDE_clares_money_l1840_184097


namespace NUMINAMATH_CALUDE_farm_legs_count_l1840_184015

/-- Calculates the total number of legs for animals in a farm -/
def total_legs (total_animals : ℕ) (chickens : ℕ) (chicken_legs : ℕ) (buffalo_legs : ℕ) : ℕ :=
  let buffalos := total_animals - chickens
  chickens * chicken_legs + buffalos * buffalo_legs

/-- Theorem: In a farm with 13 animals, where 4 are chickens and the rest are buffalos,
    the total number of animal legs is 44, given that chickens have 2 legs each and
    buffalos have 4 legs each. -/
theorem farm_legs_count :
  total_legs 13 4 2 4 = 44 := by
  sorry


end NUMINAMATH_CALUDE_farm_legs_count_l1840_184015


namespace NUMINAMATH_CALUDE_parallelogram_base_l1840_184077

theorem parallelogram_base (area height : ℝ) (h1 : area = 704) (h2 : height = 22) :
  area / height = 32 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_base_l1840_184077


namespace NUMINAMATH_CALUDE_certain_number_proof_l1840_184011

theorem certain_number_proof (k : ℤ) (x : ℝ) 
  (h1 : x * (10 : ℝ)^(k : ℝ) > 100)
  (h2 : ∀ m : ℝ, m < 4.9956356288922485 → x * (10 : ℝ)^m ≤ 100) :
  x = 0.00101 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1840_184011


namespace NUMINAMATH_CALUDE_reciprocal_and_opposite_sum_l1840_184007

theorem reciprocal_and_opposite_sum (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposite numbers
  : 3 * a * b + 2 * c + 2 * d = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_opposite_sum_l1840_184007


namespace NUMINAMATH_CALUDE_prism_volume_l1840_184086

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (a b c : ℝ) 
  (h₁ : a * b = 18) 
  (h₂ : b * c = 12) 
  (h₃ : a * c = 8) : 
  a * b * c = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1840_184086


namespace NUMINAMATH_CALUDE_det_specific_matrix_l1840_184041

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 2; 0, 6, -1; 5, -3, 1]
  Matrix.det A = -34 := by
    sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l1840_184041


namespace NUMINAMATH_CALUDE_product_value_l1840_184072

theorem product_value : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * (1 / 1024 : ℚ) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l1840_184072


namespace NUMINAMATH_CALUDE_expression_simplification_l1840_184090

theorem expression_simplification (x : ℝ) (hx : x^2 - 2*x = 0) (hx_nonzero : x ≠ 0) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1840_184090


namespace NUMINAMATH_CALUDE_probability_same_length_l1840_184037

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments in S -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of ways to choose 2 segments from S -/
def total_choices : ℕ := (total_segments.choose 2)

/-- The number of ways to choose 2 sides -/
def side_choices : ℕ := (num_sides.choose 2)

/-- The number of ways to choose 2 diagonals -/
def diagonal_choices : ℕ := (num_diagonals.choose 2)

/-- The total number of favorable outcomes (choosing two segments of the same length) -/
def favorable_outcomes : ℕ := side_choices + diagonal_choices

/-- The probability of selecting two segments of the same length from S -/
theorem probability_same_length : 
  (favorable_outcomes : ℚ) / total_choices = 17 / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_same_length_l1840_184037


namespace NUMINAMATH_CALUDE_number_of_advertisements_number_of_advertisements_proof_l1840_184028

/-- The number of advertisements shown during a race, given their duration, 
    cost per minute, and total transmission cost. -/
theorem number_of_advertisements (ad_duration : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) : ℕ :=
  5
where
  ad_duration := 3
  cost_per_minute := 4000
  total_cost := 60000

/-- Proof of the theorem -/
theorem number_of_advertisements_proof :
  number_of_advertisements 3 4000 60000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_advertisements_number_of_advertisements_proof_l1840_184028


namespace NUMINAMATH_CALUDE_product_expansion_sum_l1840_184017

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4 * x^2 - 6 * x + 5) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + 2 * c + d = -39 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l1840_184017


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l1840_184052

/-- Represents the total number of products -/
def total_products : ℕ := 5

/-- Represents the number of qualified products -/
def qualified_products : ℕ := 3

/-- Represents the number of unqualified products -/
def unqualified_products : ℕ := 2

/-- Represents the number of products randomly selected -/
def selected_products : ℕ := 2

/-- Event A: Exactly 1 unqualified product is selected -/
def event_A : Prop := sorry

/-- Event B: Exactly 2 qualified products are selected -/
def event_B : Prop := sorry

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (e1 e2 : Prop) : Prop := ¬(e1 ∧ e2)

/-- Two events are contradictory if one must occur when the other does not -/
def contradictory (e1 e2 : Prop) : Prop := (e1 ↔ ¬e2)

theorem events_mutually_exclusive_not_contradictory :
  mutually_exclusive event_A event_B ∧ ¬contradictory event_A event_B := by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l1840_184052


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l1840_184069

theorem factorial_difference_quotient : (Nat.factorial 11 - Nat.factorial 10) / Nat.factorial 9 = 100 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l1840_184069


namespace NUMINAMATH_CALUDE_second_store_earns_more_l1840_184079

/-- Represents the total value of goods sold by each department store -/
def total_goods_value : ℕ := 1000000

/-- Represents the discount rate offered by the first department store -/
def discount_rate : ℚ := 1/10

/-- Represents the number of lottery tickets given per 100 yuan spent -/
def tickets_per_hundred : ℕ := 1

/-- Represents the total number of lottery tickets -/
def total_tickets : ℕ := 10000

/-- Represents the number of first prizes -/
def first_prize_count : ℕ := 5

/-- Represents the value of each first prize -/
def first_prize_value : ℕ := 1000

/-- Represents the number of second prizes -/
def second_prize_count : ℕ := 10

/-- Represents the value of each second prize -/
def second_prize_value : ℕ := 500

/-- Represents the number of third prizes -/
def third_prize_count : ℕ := 20

/-- Represents the value of each third prize -/
def third_prize_value : ℕ := 200

/-- Represents the number of fourth prizes -/
def fourth_prize_count : ℕ := 40

/-- Represents the value of each fourth prize -/
def fourth_prize_value : ℕ := 100

/-- Represents the number of fifth prizes -/
def fifth_prize_count : ℕ := 1000

/-- Represents the value of each fifth prize -/
def fifth_prize_value : ℕ := 10

/-- Calculates the earnings of the first department store -/
def first_store_earnings : ℚ := total_goods_value * (1 - discount_rate)

/-- Calculates the total prize value for the second department store -/
def total_prize_value : ℕ := 
  first_prize_count * first_prize_value +
  second_prize_count * second_prize_value +
  third_prize_count * third_prize_value +
  fourth_prize_count * fourth_prize_value +
  fifth_prize_count * fifth_prize_value

/-- Calculates the earnings of the second department store -/
def second_store_earnings : ℕ := total_goods_value - total_prize_value

/-- Theorem stating that the second department store earns at least 72,000 yuan more than the first -/
theorem second_store_earns_more :
  (second_store_earnings : ℚ) - first_store_earnings ≥ 72000 := by
  sorry

end NUMINAMATH_CALUDE_second_store_earns_more_l1840_184079


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1840_184026

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x, k * x^2 - 3 * x + 2 = 0) → (k = 0 ∨ k = 9/8) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1840_184026


namespace NUMINAMATH_CALUDE_hard_round_points_is_five_l1840_184032

/-- A math contest with three rounds -/
structure MathContest where
  easy_correct : ℕ
  easy_points : ℕ
  avg_correct : ℕ
  avg_points : ℕ
  hard_correct : ℕ
  total_points : ℕ

/-- Kim's performance in the math contest -/
def kim_contest : MathContest := {
  easy_correct := 6
  easy_points := 2
  avg_correct := 2
  avg_points := 3
  hard_correct := 4
  total_points := 38
}

/-- Calculate the points per correct answer in the hard round -/
def hard_round_points (contest : MathContest) : ℕ :=
  (contest.total_points - (contest.easy_correct * contest.easy_points + contest.avg_correct * contest.avg_points)) / contest.hard_correct

/-- Theorem: The points per correct answer in the hard round is 5 -/
theorem hard_round_points_is_five : hard_round_points kim_contest = 5 := by
  sorry


end NUMINAMATH_CALUDE_hard_round_points_is_five_l1840_184032


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l1840_184012

theorem average_of_a_and_b (a b : ℝ) : 
  (5 + a + b) / 3 = 33 → (a + b) / 2 = 47 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l1840_184012


namespace NUMINAMATH_CALUDE_cube_operations_impossibility_l1840_184022

structure Cube :=
  (vertices : Fin 8 → ℕ)

def initial_state : Cube :=
  { vertices := λ i => if i = 0 then 1 else 0 }

def operation (c : Cube) (e : Fin 8 × Fin 8) : Cube :=
  { vertices := λ i => if i = e.1 ∨ i = e.2 then c.vertices i + 1 else c.vertices i }

def all_equal (c : Cube) : Prop :=
  ∀ i j, c.vertices i = c.vertices j

def all_divisible_by_three (c : Cube) : Prop :=
  ∀ i, c.vertices i % 3 = 0

theorem cube_operations_impossibility :
  (¬ ∃ (ops : List (Fin 8 × Fin 8)), all_equal (ops.foldl operation initial_state)) ∧
  (¬ ∃ (ops : List (Fin 8 × Fin 8)), all_divisible_by_three (ops.foldl operation initial_state)) :=
sorry

end NUMINAMATH_CALUDE_cube_operations_impossibility_l1840_184022


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1840_184014

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 139 →
  divisor = 19 →
  quotient = 7 →
  dividend = divisor * quotient + remainder →
  remainder = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1840_184014


namespace NUMINAMATH_CALUDE_spelling_bee_contestants_l1840_184064

theorem spelling_bee_contestants (initial_students : ℕ) : 
  (initial_students : ℚ) * (1 - 0.6) * (1 / 2) * (1 / 4) = 15 → 
  initial_students = 300 := by
sorry

end NUMINAMATH_CALUDE_spelling_bee_contestants_l1840_184064


namespace NUMINAMATH_CALUDE_circle_theorem_l1840_184066

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of the required circle lies
def centerLine (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the equation of the required circle
def requiredCircle (x y : ℝ) : Prop := x^2 + y^2 - x + 7*y - 32 = 0

-- Theorem statement
theorem circle_theorem :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → requiredCircle x y) ∧
    (∃ (h k : ℝ), centerLine h k ∧ 
      ∀ (x y : ℝ), requiredCircle x y ↔ (x - h)^2 + (y - k)^2 = (h - x)^2 + (k - y)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_l1840_184066


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1840_184096

def U : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0}

def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2}

def B : Set ℤ := {2, 3, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1840_184096


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l1840_184005

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {4, 5}

theorem complement_of_union_M_N :
  (M ∪ N)ᶜ = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l1840_184005


namespace NUMINAMATH_CALUDE_prime_equation_solution_l1840_184076

theorem prime_equation_solution (p q r : ℕ) (A : ℕ) 
  (h_prime_p : Nat.Prime p) 
  (h_prime_q : Nat.Prime q) 
  (h_prime_r : Nat.Prime r) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)
  (h_equation : 2*p*q*r + 50*p*q = 7*p*q*r + 55*p*r ∧ 
                7*p*q*r + 55*p*r = 8*p*q*r + 12*q*r ∧
                8*p*q*r + 12*q*r = A)
  (h_positive : A > 0) : 
  A = 1980 := by
sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l1840_184076


namespace NUMINAMATH_CALUDE_g_of_2_equals_5_l1840_184048

/-- Given a function g(x) = x^3 - 2x + 1, prove that g(2) = 5 -/
theorem g_of_2_equals_5 :
  let g : ℝ → ℝ := fun x ↦ x^3 - 2*x + 1
  g 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_equals_5_l1840_184048


namespace NUMINAMATH_CALUDE_pencils_per_box_l1840_184020

theorem pencils_per_box (total_pencils : ℕ) (num_boxes : ℕ) (pencils_per_box : ℕ) 
  (h1 : total_pencils = 27)
  (h2 : num_boxes = 3)
  (h3 : total_pencils = num_boxes * pencils_per_box) :
  pencils_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_box_l1840_184020


namespace NUMINAMATH_CALUDE_difference_of_squares_l1840_184085

theorem difference_of_squares (a b : ℝ) :
  ∃ (p q : ℝ), (a - 2*b) * (a + 2*b) = (p + q) * (p - q) ∧
                (-a + b) * (-a - b) = (p + q) * (p - q) ∧
                (-a - 1) * (1 - a) = (p + q) * (p - q) ∧
                ¬(∃ (r s : ℝ), (-x + y) * (x - y) = (r + s) * (r - s)) :=
by sorry


end NUMINAMATH_CALUDE_difference_of_squares_l1840_184085


namespace NUMINAMATH_CALUDE_hexagon_extension_l1840_184074

/-- Regular hexagon ABCDEF with side length 3 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Point Y is on the extension of AB such that AY = 2AB -/
def extend_side (h : RegularHexagon) (Y : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Y = h.A + t • (h.B - h.A) ∧ dist h.A Y = 2 * h.side_length

/-- The length of FY -/
def FY_length (h : RegularHexagon) (Y : ℝ × ℝ) : ℝ :=
  dist h.F Y

theorem hexagon_extension (h : RegularHexagon) (Y : ℝ × ℝ) 
  (ext : extend_side h Y) : FY_length h Y = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_extension_l1840_184074


namespace NUMINAMATH_CALUDE_parabola_p_value_l1840_184075

/-- Given a parabola y^2 = 2px with directrix x = -2, prove that p = 4 -/
theorem parabola_p_value (y x p : ℝ) : 
  (y^2 = 2*p*x) → -- Parabola equation
  (-p/2 = -2) →   -- Directrix equation (transformed)
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_p_value_l1840_184075


namespace NUMINAMATH_CALUDE_triangle_base_length_l1840_184070

theorem triangle_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 25) 
  (h2 : height = 5) : 
  area = (base * height) / 2 → base = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l1840_184070


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l1840_184098

theorem complex_expression_evaluation : 
  (0.027)^(-1/3 : ℝ) - (-1/7)^(-2 : ℝ) + (25/9 : ℝ)^(1/2 : ℝ) - (Real.sqrt 2 - 1)^(0 : ℝ) = -45 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l1840_184098


namespace NUMINAMATH_CALUDE_range_of_a_given_proposition_l1840_184089

theorem range_of_a_given_proposition (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a ≥ 0) → a ≥ -8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_proposition_l1840_184089


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_is_625_div_3_l1840_184049

/-- A right triangle XYZ in the xy-plane with specific properties -/
structure RightTriangle where
  /-- The length of the hypotenuse XY -/
  hypotenuse_length : ℝ
  /-- The y-intercept of the line containing the median through X -/
  median_x_intercept : ℝ
  /-- The slope of the line containing the median through Y -/
  median_y_slope : ℝ
  /-- The y-intercept of the line containing the median through Y -/
  median_y_intercept : ℝ
  /-- Condition: The hypotenuse length is 50 -/
  hypotenuse_cond : hypotenuse_length = 50
  /-- Condition: The median through X lies on y = x + 5 -/
  median_x_cond : median_x_intercept = 5
  /-- Condition: The median through Y lies on y = 3x + 6 -/
  median_y_cond : median_y_slope = 3 ∧ median_y_intercept = 6

/-- The theorem stating that the area of the specific right triangle is 625/3 -/
theorem right_triangle_area (t : RightTriangle) : ℝ :=
  625 / 3

/-- The main theorem to be proved -/
theorem right_triangle_area_is_625_div_3 (t : RightTriangle) :
  right_triangle_area t = 625 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_is_625_div_3_l1840_184049


namespace NUMINAMATH_CALUDE_exam_mean_score_l1840_184045

theorem exam_mean_score (a b : ℕ) (mean_a mean_b : ℝ) :
  a > 0 ∧ b > 0 →
  mean_a = 90 →
  mean_b = 78 →
  a = (5 : ℝ) / 7 * b →
  ∃ (max_score_a : ℝ), max_score_a = 100 ∧ max_score_a ≥ mean_b + 20 →
  (mean_a * a + mean_b * b) / (a + b) = 83 :=
by sorry

end NUMINAMATH_CALUDE_exam_mean_score_l1840_184045


namespace NUMINAMATH_CALUDE_original_room_population_l1840_184078

theorem original_room_population (x : ℚ) : 
  (1 / 2 : ℚ) * x = 18 →
  (2 / 3 : ℚ) * x - (1 / 4 : ℚ) * ((2 / 3 : ℚ) * x) = 18 →
  x = 36 := by sorry

end NUMINAMATH_CALUDE_original_room_population_l1840_184078


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1840_184094

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Define the inequality
def inequality (x : ℝ) : Prop := (2 - x) / (x + 1) ≥ 0

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1840_184094


namespace NUMINAMATH_CALUDE_max_correct_answers_l1840_184039

theorem max_correct_answers (total_questions : ℕ) (score : ℤ) : 
  total_questions = 25 → score = 65 → 
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    4 * correct - incorrect = score ∧
    correct ≤ 18 ∧
    ∀ c i u : ℕ, 
      c + i + u = total_questions → 
      4 * c - i = score → 
      c ≤ correct :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1840_184039


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1840_184016

/-- The line equation passing through a fixed point for all values of k -/
def line_equation (k x y : ℝ) : Prop :=
  (k + 2) * x + (1 - k) * y - 4 * k - 5 = 0

/-- Theorem stating that the line passes through the point (3, -1) for all k -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k 3 (-1) :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1840_184016


namespace NUMINAMATH_CALUDE_sum_greater_than_one_l1840_184038

theorem sum_greater_than_one : 
  (let a := [1/4, 2/8, 3/4]
   let b := [3, -1.5, -0.5]
   let c := [0.25, 0.75, 0.05]
   let d := [3/2, -3/4, 1/4]
   let e := [1.5, 1.5, -2]
   (a.sum > 1 ∧ c.sum > 1) ∧
   (b.sum ≤ 1 ∧ d.sum ≤ 1 ∧ e.sum ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_one_l1840_184038


namespace NUMINAMATH_CALUDE_water_jars_problem_l1840_184046

theorem water_jars_problem (S L : ℝ) (h1 : S > 0) (h2 : L > 0) (h3 : S < L) :
  let water_amount := S * (1/3)
  (water_amount = L * (1/2)) →
  (L * (1/2) + water_amount) / L = 1 := by
sorry

end NUMINAMATH_CALUDE_water_jars_problem_l1840_184046


namespace NUMINAMATH_CALUDE_incenter_vector_ratio_l1840_184023

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Main theorem
theorem incenter_vector_ratio (t : Triangle) 
  (h1 : dist t.A t.B = 6)
  (h2 : dist t.B t.C = 7)
  (h3 : dist t.A t.C = 4)
  (O : ℝ × ℝ)
  (hO : O = incenter t)
  (p q : ℝ)
  (h4 : vec_add (vec_scale (-1) O) t.A = vec_add (vec_scale p (vec_add (vec_scale (-1) t.A) t.B)) 
                                                 (vec_scale q (vec_add (vec_scale (-1) t.A) t.C)))
  : p / q = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_incenter_vector_ratio_l1840_184023


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l1840_184008

theorem product_of_roots_plus_one (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (1 + p) * (1 + q) * (1 + r) = 51 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l1840_184008


namespace NUMINAMATH_CALUDE_mika_initial_stickers_l1840_184044

/-- The number of stickers Mika had initially -/
def initial_stickers : ℕ := 26

/-- The number of stickers Mika bought -/
def bought_stickers : ℕ := 26

/-- The number of stickers Mika got for her birthday -/
def birthday_stickers : ℕ := 20

/-- The number of stickers Mika gave to her sister -/
def given_stickers : ℕ := 6

/-- The number of stickers Mika used for the greeting card -/
def used_stickers : ℕ := 58

/-- The number of stickers Mika is left with -/
def remaining_stickers : ℕ := 2

theorem mika_initial_stickers :
  initial_stickers + bought_stickers + birthday_stickers - given_stickers - used_stickers = remaining_stickers :=
by
  sorry

#check mika_initial_stickers

end NUMINAMATH_CALUDE_mika_initial_stickers_l1840_184044


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_3_l1840_184018

theorem smallest_lcm_with_gcd_3 (k l : ℕ) : 
  k ≥ 1000 ∧ k ≤ 9999 ∧ l ≥ 1000 ∧ l ≤ 9999 ∧ Nat.gcd k l = 3 →
  Nat.lcm k l ≥ 335670 ∧ ∃ (k₀ l₀ : ℕ), k₀ ≥ 1000 ∧ k₀ ≤ 9999 ∧ l₀ ≥ 1000 ∧ l₀ ≤ 9999 ∧ 
  Nat.gcd k₀ l₀ = 3 ∧ Nat.lcm k₀ l₀ = 335670 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_3_l1840_184018


namespace NUMINAMATH_CALUDE_drive_duration_proof_l1840_184013

def podcast1_duration : ℕ := 45
def podcast2_duration : ℕ := 2 * podcast1_duration
def podcast3_duration : ℕ := 105
def podcast4_duration : ℕ := 60
def podcast5_duration : ℕ := 60

def total_duration : ℕ := podcast1_duration + podcast2_duration + podcast3_duration + podcast4_duration + podcast5_duration

theorem drive_duration_proof :
  total_duration / 60 = 6 := by sorry

end NUMINAMATH_CALUDE_drive_duration_proof_l1840_184013


namespace NUMINAMATH_CALUDE_venue_cost_venue_cost_is_10000_l1840_184095

/-- Calculates the venue cost for John's wedding --/
theorem venue_cost (cost_per_guest : ℕ) (john_guests : ℕ) (wife_extra_percent : ℕ) (total_cost : ℕ) : ℕ :=
  let wife_guests := john_guests + (wife_extra_percent * john_guests) / 100
  let guest_cost := cost_per_guest * wife_guests
  total_cost - guest_cost

/-- Proves that the venue cost is $10,000 given the specified conditions --/
theorem venue_cost_is_10000 :
  venue_cost 500 50 60 50000 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_venue_cost_venue_cost_is_10000_l1840_184095


namespace NUMINAMATH_CALUDE_min_isosceles_right_triangles_10x100_l1840_184006

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle

/-- Returns the minimum number of isosceles right triangles needed to cover a rectangle -/
def minIsoscelesRightTriangles (r : Rectangle) : ℕ := sorry

/-- The theorem statement -/
theorem min_isosceles_right_triangles_10x100 :
  minIsoscelesRightTriangles ⟨100, 10⟩ = 11 := by sorry

end NUMINAMATH_CALUDE_min_isosceles_right_triangles_10x100_l1840_184006


namespace NUMINAMATH_CALUDE_certain_number_problem_l1840_184010

theorem certain_number_problem (x : ℝ) (certain_number : ℝ) 
  (h1 : certain_number * x = 675)
  (h2 : x = 27) : 
  certain_number = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1840_184010


namespace NUMINAMATH_CALUDE_correct_count_of_students_using_both_colors_l1840_184073

/-- The number of students using both green and red colors in a painting activity. -/
def students_using_both_colors (total_students green_users red_users : ℕ) : ℕ :=
  green_users + red_users - total_students

/-- Theorem stating that the number of students using both colors is correct. -/
theorem correct_count_of_students_using_both_colors
  (total_students green_users red_users : ℕ)
  (h1 : total_students = 70)
  (h2 : green_users = 52)
  (h3 : red_users = 56) :
  students_using_both_colors total_students green_users red_users = 38 := by
  sorry

end NUMINAMATH_CALUDE_correct_count_of_students_using_both_colors_l1840_184073


namespace NUMINAMATH_CALUDE_exists_irrational_in_interval_l1840_184021

theorem exists_irrational_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 0.3 0.4 ∧ Irrational x ∧ x * (x + 1) * (x + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_in_interval_l1840_184021


namespace NUMINAMATH_CALUDE_sandwich_interval_is_40_minutes_l1840_184033

/-- Represents the Sandwich Shop's operations -/
structure SandwichShop where
  hours_per_day : ℕ
  peppers_per_day : ℕ
  peppers_per_sandwich : ℕ

/-- Calculates the interval between sandwiches in minutes -/
def sandwich_interval (shop : SandwichShop) : ℕ :=
  let sandwiches_per_day := shop.peppers_per_day / shop.peppers_per_sandwich
  let minutes_per_day := shop.hours_per_day * 60
  minutes_per_day / sandwiches_per_day

/-- The theorem stating the interval between sandwiches is 40 minutes -/
theorem sandwich_interval_is_40_minutes :
  ∀ (shop : SandwichShop),
    shop.hours_per_day = 8 →
    shop.peppers_per_day = 48 →
    shop.peppers_per_sandwich = 4 →
    sandwich_interval shop = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_sandwich_interval_is_40_minutes_l1840_184033


namespace NUMINAMATH_CALUDE_evaluate_expression_l1840_184084

theorem evaluate_expression : (2^2010 * 3^2012) / 6^2011 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1840_184084


namespace NUMINAMATH_CALUDE_reflection_result_l1840_184057

/-- Reflects a point across the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point across the line y = -x + 2 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 2)  -- Translate down by 2
  let p'' := (-p'.2, -p'.1) -- Reflect across y = -x
  (p''.1, p''.2 + 2)        -- Translate back up by 2

/-- The final position of point R after two reflections -/
def R_final : ℝ × ℝ :=
  reflect_line (reflect_x_axis (6, 1))

theorem reflection_result :
  R_final = (-3, -4) :=
by sorry

end NUMINAMATH_CALUDE_reflection_result_l1840_184057


namespace NUMINAMATH_CALUDE_factors_of_given_number_l1840_184040

/-- The number of distinct natural-number factors of 4^5 · 5^3 · 7^2 -/
def num_factors : ℕ := 132

/-- The given number -/
def given_number : ℕ := 4^5 * 5^3 * 7^2

/-- A function that counts the number of distinct natural-number factors of a given natural number -/
def count_factors (n : ℕ) : ℕ := sorry

theorem factors_of_given_number :
  count_factors given_number = num_factors := by sorry

end NUMINAMATH_CALUDE_factors_of_given_number_l1840_184040


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l1840_184082

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Given vectors a and b, prove that if they are collinear, then y = -2 -/
theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (-3, 1)
  let b : ℝ × ℝ := (6, y)
  collinear a b → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l1840_184082


namespace NUMINAMATH_CALUDE_sin_30_degrees_l1840_184019

/-- Sine of 30 degrees is equal to 1/2 -/
theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l1840_184019


namespace NUMINAMATH_CALUDE_inequality_problem_l1840_184047

theorem inequality_problem (a b c d : ℝ) 
  (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l1840_184047


namespace NUMINAMATH_CALUDE_blast_distance_problem_l1840_184068

/-- The distance traveled by sound in a given time -/
def sound_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The problem statement -/
theorem blast_distance_problem (man_speed : ℝ) (sound_speed : ℝ) (total_time : ℝ) (blast_interval : ℝ) :
  sound_speed = 330 →
  total_time = 30 * 60 + 12 →
  blast_interval = 30 * 60 →
  sound_distance sound_speed (total_time - blast_interval) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_blast_distance_problem_l1840_184068


namespace NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l1840_184099

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height circle_circumference : ℝ),
  rectangle_width = 9 →
  rectangle_height = 12 →
  circle_circumference = π * (rectangle_width^2 + rectangle_height^2).sqrt →
  circle_circumference = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_circle_circumference_l1840_184099


namespace NUMINAMATH_CALUDE_equation_solution_l1840_184000

theorem equation_solution : 
  ∃ x : ℚ, (5 * x - 2) / (6 * x - 6) = 3 / 4 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1840_184000


namespace NUMINAMATH_CALUDE_B_power_106_l1840_184051

def B : Matrix (Fin 3) (Fin 3) ℤ := ![![0, 1, 0], ![0, 0, -1], ![0, 1, 0]]

theorem B_power_106 : B^106 = ![![0, 0, -1], ![0, -1, 0], ![0, 0, -1]] := by sorry

end NUMINAMATH_CALUDE_B_power_106_l1840_184051


namespace NUMINAMATH_CALUDE_circle_and_tangents_l1840_184030

-- Define the points
def A : ℝ × ℝ := (-1, -3)
def B : ℝ × ℝ := (5, 5)
def M : ℝ × ℝ := (-3, 2)

-- Define the circle O
def O : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 25}

-- Define the tangent lines
def tangent_lines : Set (Set (ℝ × ℝ)) := 
  {{p | p.1 = -3}, {p | 12 * p.1 - 5 * p.2 + 46 = 0}}

theorem circle_and_tangents :
  (∀ p ∈ O, (p.1 - 2)^2 + (p.2 - 1)^2 = 25) ∧
  (∀ l ∈ tangent_lines, ∃ p ∈ O, p ∈ l ∧ 
    (∀ q ∈ O, q ≠ p → q ∉ l)) ∧
  (∀ l ∈ tangent_lines, M ∈ l) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangents_l1840_184030


namespace NUMINAMATH_CALUDE_point_B_in_first_quadrant_l1840_184083

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the first quadrant -/
def is_in_first_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The theorem to be proved -/
theorem point_B_in_first_quadrant (A : Point2D) (hA : is_in_second_quadrant A) :
  let B : Point2D := ⟨-2 * A.x, (1/3) * A.y⟩
  is_in_first_quadrant B := by
  sorry

end NUMINAMATH_CALUDE_point_B_in_first_quadrant_l1840_184083


namespace NUMINAMATH_CALUDE_parking_lot_problem_l1840_184002

theorem parking_lot_problem (total_vehicles : ℕ) (total_wheels : ℕ) 
  (h1 : total_vehicles = 24) 
  (h2 : total_wheels = 86) : 
  ∃ (cars motorcycles : ℕ), 
    cars + motorcycles = total_vehicles ∧ 
    4 * cars + 3 * motorcycles = total_wheels ∧ 
    motorcycles = 10 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l1840_184002


namespace NUMINAMATH_CALUDE_gcd_of_g_103_104_l1840_184059

/-- The function g as defined in the problem -/
def g (x : ℤ) : ℤ := x^2 - x + 2025

/-- The theorem stating that the GCD of g(103) and g(104) is 2 -/
theorem gcd_of_g_103_104 : Int.gcd (g 103) (g 104) = 2 := by sorry

end NUMINAMATH_CALUDE_gcd_of_g_103_104_l1840_184059


namespace NUMINAMATH_CALUDE_total_rainfall_is_23_l1840_184055

/-- Rainfall data for three days --/
structure RainfallData :=
  (monday_hours : ℕ)
  (monday_rate : ℕ)
  (tuesday_hours : ℕ)
  (tuesday_rate : ℕ)
  (wednesday_hours : ℕ)

/-- Calculate total rainfall for three days --/
def total_rainfall (data : RainfallData) : ℕ :=
  data.monday_hours * data.monday_rate +
  data.tuesday_hours * data.tuesday_rate +
  data.wednesday_hours * (2 * data.tuesday_rate)

/-- Theorem: The total rainfall for the given conditions is 23 inches --/
theorem total_rainfall_is_23 (data : RainfallData)
  (h1 : data.monday_hours = 7)
  (h2 : data.monday_rate = 1)
  (h3 : data.tuesday_hours = 4)
  (h4 : data.tuesday_rate = 2)
  (h5 : data.wednesday_hours = 2) :
  total_rainfall data = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_is_23_l1840_184055


namespace NUMINAMATH_CALUDE_point_coordinates_l1840_184043

theorem point_coordinates (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : y = 2) (h4 : x = 4) :
  (x, y) = (4, 2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1840_184043


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l1840_184042

/-- A right pyramid with a square base -/
structure RightPyramid where
  base_area : ℝ
  total_surface_area : ℝ
  triangular_face_area : ℝ

/-- The volume of a right pyramid with a square base -/
def volume (p : RightPyramid) : ℝ := sorry

/-- Theorem: The volume of the specific right pyramid is 310.5√207 cubic units -/
theorem specific_pyramid_volume :
  ∀ (p : RightPyramid),
    p.total_surface_area = 486 ∧
    p.triangular_face_area = p.base_area / 3 →
    volume p = 310.5 * Real.sqrt 207 :=
by sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l1840_184042


namespace NUMINAMATH_CALUDE_largest_power_dividing_product_l1840_184092

def pow (n : ℕ) : ℕ :=
  sorry

def product_pow : ℕ :=
  sorry

theorem largest_power_dividing_product :
  (∃ m : ℕ, (2310 : ℕ)^m ∣ product_pow ∧ 
    ∀ k : ℕ, (2310 : ℕ)^k ∣ product_pow → k ≤ m) ∧
  (∃ m : ℕ, (2310 : ℕ)^m ∣ product_pow ∧ m = 319) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_product_l1840_184092


namespace NUMINAMATH_CALUDE_average_of_r_s_t_l1840_184024

theorem average_of_r_s_t (r s t : ℝ) (h : (5 / 2) * (r + s + t) = 25) :
  (r + s + t) / 3 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_r_s_t_l1840_184024


namespace NUMINAMATH_CALUDE_total_amount_paid_l1840_184036

/-- Represents the purchase of a fruit with its quantity and price per kg -/
structure FruitPurchase where
  quantity : ℕ
  price_per_kg : ℕ

/-- Calculates the total cost of a fruit purchase -/
def total_cost (purchase : FruitPurchase) : ℕ :=
  purchase.quantity * purchase.price_per_kg

/-- Represents Tom's fruit shopping -/
def fruit_shopping : List FruitPurchase :=
  [
    { quantity := 8, price_per_kg := 70 },  -- Apples
    { quantity := 9, price_per_kg := 65 },  -- Mangoes
    { quantity := 5, price_per_kg := 50 },  -- Oranges
    { quantity := 3, price_per_kg := 30 }   -- Bananas
  ]

/-- Theorem: The total amount Tom paid for all fruits is $1485 -/
theorem total_amount_paid : (fruit_shopping.map total_cost).sum = 1485 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1840_184036


namespace NUMINAMATH_CALUDE_sum_abs_coefficients_f6_l1840_184087

def polynomial_sequence : ℕ → (ℝ → ℝ) 
  | 0 => λ x => 1
  | n + 1 => λ x => (x^2 - 1) * (polynomial_sequence n x) - 2*x

def sum_abs_coefficients (f : ℝ → ℝ) : ℝ := sorry

theorem sum_abs_coefficients_f6 : 
  sum_abs_coefficients (polynomial_sequence 6) = 190 := by sorry

end NUMINAMATH_CALUDE_sum_abs_coefficients_f6_l1840_184087


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1840_184001

theorem smallest_five_digit_divisible_by_first_five_primes : 
  ∃ n : ℕ, 
    n ≥ 10000 ∧ 
    n < 100000 ∧ 
    2 ∣ n ∧ 
    3 ∣ n ∧ 
    5 ∣ n ∧ 
    7 ∣ n ∧ 
    11 ∣ n ∧ 
    ∀ m : ℕ, 
      m ≥ 10000 ∧ 
      m < 100000 ∧ 
      2 ∣ m ∧ 
      3 ∣ m ∧ 
      5 ∣ m ∧ 
      7 ∣ m ∧ 
      11 ∣ m → 
      n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1840_184001


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1840_184067

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (h_arithmetic : ArithmeticSequence a)
  (h_eq : a 3 + a 9 = 4 * a 5)
  (h_a2 : a 2 = -8) :
  ∃ d : ℤ, d = 4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1840_184067


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l1840_184029

theorem smallest_x_for_equation : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∃ (y : ℕ), y > 0 ∧ (3 : ℚ) / 4 = y / (210 + x)) ∧
  (∀ (x' : ℕ), x' > 0 → x' < x → 
    ¬∃ (y : ℕ), y > 0 ∧ (3 : ℚ) / 4 = y / (210 + x')) ∧
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l1840_184029


namespace NUMINAMATH_CALUDE_parabola_sum_l1840_184054

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Check if a point is on the parabola -/
def contains_point (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Check if the parabola has a vertical axis of symmetry -/
def has_vertical_axis (p : Parabola) : Prop := sorry

theorem parabola_sum (p : Parabola) :
  vertex p = (3, 7) →
  has_vertical_axis p →
  contains_point p 0 4 →
  p.a + p.b + p.c = 5.666 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l1840_184054


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l1840_184050

/-- A trapezoid with angles in arithmetic progression -/
structure ArithmeticTrapezoid where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference
  angle_sum : a + (a + d) + (a + 2*d) + (a + 3*d) = 360
  largest_angle : a + 3*d = 150

theorem smallest_angle_measure (t : ArithmeticTrapezoid) : t.a = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l1840_184050


namespace NUMINAMATH_CALUDE_sample_size_b_l1840_184062

/-- Represents the number of products in each batch -/
structure BatchSizes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sample sizes from each batch -/
structure SampleSizes where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The theorem to prove -/
theorem sample_size_b (batchSizes : BatchSizes) (sampleSizes : SampleSizes) : 
  batchSizes.a + batchSizes.b + batchSizes.c = 210 →
  batchSizes.c - batchSizes.b = batchSizes.b - batchSizes.a →
  sampleSizes.a + sampleSizes.b + sampleSizes.c = 60 →
  sampleSizes.c - sampleSizes.b = sampleSizes.b - sampleSizes.a →
  sampleSizes.b = 20 := by
sorry

end NUMINAMATH_CALUDE_sample_size_b_l1840_184062


namespace NUMINAMATH_CALUDE_same_color_probability_l1840_184091

-- Define the number of red and blue plates
def red_plates : ℕ := 5
def blue_plates : ℕ := 4

-- Define the total number of plates
def total_plates : ℕ := red_plates + blue_plates

-- Define the function to calculate combinations
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem statement
theorem same_color_probability :
  (choose red_plates 2 + choose blue_plates 2) / choose total_plates 2 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1840_184091


namespace NUMINAMATH_CALUDE_theater_occupancy_l1840_184071

theorem theater_occupancy (total_seats empty_seats : ℕ) 
  (h1 : total_seats = 750) 
  (h2 : empty_seats = 218) : 
  total_seats - empty_seats = 532 := by
  sorry

end NUMINAMATH_CALUDE_theater_occupancy_l1840_184071


namespace NUMINAMATH_CALUDE_smartphone_savings_theorem_l1840_184034

/-- The amount saved per month in yuan -/
def monthly_savings : ℕ := 530

/-- The cost of the smartphone in yuan -/
def smartphone_cost : ℕ := 2000

/-- The number of months required to save for the smartphone -/
def months_required : ℕ := 4

theorem smartphone_savings_theorem : 
  monthly_savings * months_required ≥ smartphone_cost :=
sorry

end NUMINAMATH_CALUDE_smartphone_savings_theorem_l1840_184034
