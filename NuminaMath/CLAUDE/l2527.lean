import Mathlib

namespace walmart_complaints_l2527_252757

/-- The number of complaints received by a Walmart store over a period of days --/
def total_complaints (normal_rate : ℝ) (short_staffed_factor : ℝ) (checkout_broken_factor : ℝ) (days : ℝ) : ℝ :=
  normal_rate * short_staffed_factor * checkout_broken_factor * days

/-- Theorem stating that under given conditions, the total complaints over 3 days is 576 --/
theorem walmart_complaints :
  total_complaints 120 (4/3) 1.2 3 = 576 := by
  sorry

end walmart_complaints_l2527_252757


namespace rectangle_inside_circle_l2527_252701

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point on the unit circle
def point_on_circle (p : ℝ × ℝ) : Prop :=
  unit_circle p.1 p.2

-- Define a point inside the unit circle
def point_inside_circle (q : ℝ × ℝ) : Prop :=
  q.1^2 + q.2^2 < 1

-- Define the rectangle with diagonal pq and sides parallel to axes
def rectangle_with_diagonal (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r | p.1 ≤ r.1 ∧ r.1 ≤ q.1 ∧ q.2 ≤ r.2 ∧ r.2 ≤ p.2 ∨
       q.1 ≤ r.1 ∧ r.1 ≤ p.1 ∧ p.2 ≤ r.2 ∧ r.2 ≤ q.2}

-- Theorem statement
theorem rectangle_inside_circle (p q : ℝ × ℝ) :
  point_on_circle p → point_inside_circle q →
  ∀ r ∈ rectangle_with_diagonal p q, r.1^2 + r.2^2 ≤ 1 :=
by sorry

end rectangle_inside_circle_l2527_252701


namespace min_value_fraction_l2527_252720

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2*x + y = 1) :
  (x^2 + y^2 + x) / (x*y) ≥ 2*Real.sqrt 3 + 1 :=
by sorry

end min_value_fraction_l2527_252720


namespace marks_used_days_ratio_l2527_252779

def total_allotted_days : ℕ := 20
def hours_per_day : ℕ := 8
def unused_hours : ℕ := 80

theorem marks_used_days_ratio :
  let unused_days : ℕ := unused_hours / hours_per_day
  let used_days : ℕ := total_allotted_days - unused_days
  (used_days : ℚ) / total_allotted_days = 1 / 2 := by sorry

end marks_used_days_ratio_l2527_252779


namespace max_value_sqrt_sum_l2527_252769

theorem max_value_sqrt_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1 := by
  sorry

end max_value_sqrt_sum_l2527_252769


namespace quadratic_incenter_theorem_l2527_252755

/-- A quadratic function that intersects the coordinate axes at three points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- A triangle formed by the intersection points of a quadratic function with the coordinate axes -/
structure IntersectionTriangle where
  quad : QuadraticFunction
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle -/
def incenter (t : IntersectionTriangle) : ℝ × ℝ := sorry

/-- The theorem statement -/
theorem quadratic_incenter_theorem (t : IntersectionTriangle) 
  (h1 : t.A.2 = 0 ∨ t.A.1 = 0)
  (h2 : t.B.2 = 0 ∨ t.B.1 = 0)
  (h3 : t.C.2 = 0 ∨ t.C.1 = 0)
  (h4 : t.A ≠ t.B ∧ t.B ≠ t.C ∧ t.A ≠ t.C)
  (h5 : ∃ (x : ℝ), incenter t = (x, x)) :
  t.quad.a + t.quad.b + 1 = 0 := by
  sorry

end quadratic_incenter_theorem_l2527_252755


namespace inequality_proof_l2527_252784

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, f 1 x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  1/m + 1/(2*n) = 1 →
  m + 2*n ≥ 4 := by
sorry

end inequality_proof_l2527_252784


namespace stratified_sampling_proportion_l2527_252706

theorem stratified_sampling_proportion (total : ℕ) (males : ℕ) (females_selected : ℕ) :
  total = 220 →
  males = 60 →
  females_selected = 32 →
  (males / total : ℚ) = ((12 : ℕ) / (12 + females_selected) : ℚ) :=
by sorry

end stratified_sampling_proportion_l2527_252706


namespace f_continuous_at_2_l2527_252740

def f (x : ℝ) := -2 * x^2 - 5

theorem f_continuous_at_2 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by sorry

end f_continuous_at_2_l2527_252740


namespace sqrt_three_squared_l2527_252761

theorem sqrt_three_squared : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end sqrt_three_squared_l2527_252761


namespace calculator_mistake_l2527_252721

theorem calculator_mistake (x : ℝ) (h : Real.sqrt x = 9) : x^2 = 6561 := by
  sorry

end calculator_mistake_l2527_252721


namespace ball_probabilities_l2527_252788

/-- Represents a box of balls -/
structure Box where
  red : ℕ
  blue : ℕ

/-- The initial state of Box A -/
def box_a : Box := ⟨2, 4⟩

/-- The initial state of Box B -/
def box_b : Box := ⟨3, 3⟩

/-- The number of balls drawn from Box A -/
def balls_drawn : ℕ := 2

/-- Probability of drawing at least one blue ball from Box A -/
def prob_blue_from_a : ℚ := 14/15

/-- Probability of drawing a blue ball from Box B after transfer -/
def prob_blue_from_b : ℚ := 13/24

theorem ball_probabilities :
  (prob_blue_from_a = 14/15) ∧ (prob_blue_from_b = 13/24) := by sorry

end ball_probabilities_l2527_252788


namespace perpendicular_line_to_plane_l2527_252728

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp : Plane → Plane → Prop)
variable (perp_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem perpendicular_line_to_plane 
  (α β γ : Plane) (m l : Line) 
  (h1 : perp α γ)
  (h2 : intersect γ α = m)
  (h3 : intersect γ β = l)
  (h4 : perp_line l m) :
  perp_line_plane l α :=
sorry

end perpendicular_line_to_plane_l2527_252728


namespace a_range_l2527_252793

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≤ 0

-- Define the theorem
theorem a_range (a : ℝ) : 
  (a > 0) →
  (∀ x, ¬(p x) → ¬(q x a)) →
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ≥ 9 :=
sorry

end a_range_l2527_252793


namespace triangle_properties_l2527_252786

theorem triangle_properties (a b c A B C : ℝ) (r : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Condition: √3b = a(√3cosC - sinC)
  (Real.sqrt 3 * b = a * (Real.sqrt 3 * Real.cos C - Real.sin C)) →
  -- Condition: a = 8
  (a = 8) →
  -- Condition: Radius of incircle = √3
  (r = Real.sqrt 3) →
  -- Proof that angle A = 2π/3
  (A = 2 * Real.pi / 3) ∧
  -- Proof that perimeter = 18
  (a + b + c = 18) :=
by sorry

end triangle_properties_l2527_252786


namespace parrot_silence_explanation_l2527_252737

-- Define the parrot type
structure Parrot where
  repeats_heard_words : Bool
  is_silent : Bool

-- Define the environment
structure Environment where
  words_spoken : Bool

-- Define the theorem
theorem parrot_silence_explanation (p : Parrot) (e : Environment) :
  p.repeats_heard_words ∧ p.is_silent →
  (¬e.words_spoken ∨ ¬p.repeats_heard_words) :=
by
  sorry

-- The negation of repeats_heard_words represents deafness

end parrot_silence_explanation_l2527_252737


namespace triangle_max_area_l2527_252767

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    prove that the maximum area is 9√7 when a = 6 and √7 * b * cos(A) = 3 * a * sin(B) -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 6 →
  Real.sqrt 7 * b * Real.cos A = 3 * a * Real.sin B →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ S ≤ 9 * Real.sqrt 7) :=
sorry

end triangle_max_area_l2527_252767


namespace square_sum_product_l2527_252732

theorem square_sum_product (a b : ℝ) (h1 : a + b = -3) (h2 : a * b = 2) :
  a^2 * b + a * b^2 = -6 := by
  sorry

end square_sum_product_l2527_252732


namespace f_derivative_at_2014_l2527_252747

noncomputable def f (f'2014 : ℝ) : ℝ → ℝ := 
  λ x => (1/2) * x^2 + 2 * x * f'2014 + 2014 * Real.log x

theorem f_derivative_at_2014 : 
  ∃ f'2014 : ℝ, (deriv (f f'2014)) 2014 = -2015 := by sorry

end f_derivative_at_2014_l2527_252747


namespace dog_spots_l2527_252789

/-- The number of spots on dogs problem -/
theorem dog_spots (rover_spots : ℕ) (cisco_spots : ℕ) (granger_spots : ℕ)
  (h1 : rover_spots = 46)
  (h2 : cisco_spots = rover_spots / 2 - 5)
  (h3 : granger_spots = 5 * cisco_spots) :
  granger_spots + cisco_spots = 108 := by
  sorry

end dog_spots_l2527_252789


namespace interval_of_decrease_f_left_endpoint_neg_infinity_right_endpoint_is_one_l2527_252751

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Theorem stating the interval of decrease
theorem interval_of_decrease_f :
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f x > f y :=
by sorry

-- The left endpoint of the interval is negative infinity
theorem left_endpoint_neg_infinity :
  ∀ M : ℝ, ∃ x : ℝ, x < M ∧ ∀ y : ℝ, x < y ∧ y ≤ 1 → f x > f y :=
by sorry

-- The right endpoint of the interval is 1
theorem right_endpoint_is_one :
  ∀ ε > 0, ∃ x : ℝ, 1 < x ∧ x < 1 + ε ∧ f 1 < f x :=
by sorry

end interval_of_decrease_f_left_endpoint_neg_infinity_right_endpoint_is_one_l2527_252751


namespace sphere_volume_increase_l2527_252746

/-- The volume of a sphere increases by a factor of 8 when its radius is doubled -/
theorem sphere_volume_increase (r : ℝ) (hr : r > 0) : 
  (4 / 3 * Real.pi * (2 * r)^3) / (4 / 3 * Real.pi * r^3) = 8 := by
  sorry

end sphere_volume_increase_l2527_252746


namespace not_p_or_q_false_implies_at_least_one_true_l2527_252750

theorem not_p_or_q_false_implies_at_least_one_true (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
  sorry

end not_p_or_q_false_implies_at_least_one_true_l2527_252750


namespace smallest_integer_bound_l2527_252777

theorem smallest_integer_bound (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧  -- Largest is 90
  (a + b + c + d) / 4 = 70  -- Average is 70
  → a ≥ 13 := by
sorry

end smallest_integer_bound_l2527_252777


namespace f_5_equals_142_l2527_252715

-- Define the function f
def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

-- Theorem statement
theorem f_5_equals_142 :
  ∃ y : ℝ, f 2 y = 100 → f 5 y = 142 :=
by
  sorry

end f_5_equals_142_l2527_252715


namespace simplify_sqrt_sum_l2527_252736

theorem simplify_sqrt_sum : 
  Real.sqrt (10 + 6 * Real.sqrt 3) + Real.sqrt (10 - 6 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
sorry

end simplify_sqrt_sum_l2527_252736


namespace right_triangle_with_three_isosceles_l2527_252749

/-- A right-angled triangle that can be divided into three isosceles triangles has acute angles of 22.5° and 67.5°. -/
theorem right_triangle_with_three_isosceles (α β : Real) : 
  α + β = 90 → -- The sum of acute angles in a right triangle is 90°
  (∃ (γ : Real), γ = 90 ∧ 2*α + 2*α = γ) → -- One of the isosceles triangles has a right angle and two equal angles of 2α
  (α = 22.5 ∧ β = 67.5) := by
  sorry


end right_triangle_with_three_isosceles_l2527_252749


namespace sector_area_of_circle_l2527_252780

/-- Given a circle with circumference 16π, prove that the area of a sector
    subtending a central angle of 90° is 16π. -/
theorem sector_area_of_circle (C : ℝ) (θ : ℝ) (h1 : C = 16 * Real.pi) (h2 : θ = 90) :
  let r := C / (2 * Real.pi)
  let A := Real.pi * r^2
  let sector_area := (θ / 360) * A
  sector_area = 16 * Real.pi :=
by sorry

end sector_area_of_circle_l2527_252780


namespace children_tickets_sold_l2527_252733

theorem children_tickets_sold (adult_price child_price total_tickets total_revenue : ℚ)
  (h1 : adult_price = 6)
  (h2 : child_price = 9/2)
  (h3 : total_tickets = 400)
  (h4 : total_revenue = 2100)
  (h5 : ∃ (adult_tickets child_tickets : ℚ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue) :
  ∃ (child_tickets : ℚ), child_tickets = 200 := by
  sorry

end children_tickets_sold_l2527_252733


namespace train_passing_time_l2527_252735

/-- Proves that a train of given length and speed takes a specific time to pass a stationary object. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 150 →
  train_speed_kmh = 36 →
  passing_time = 15 →
  passing_time = train_length / (train_speed_kmh * (5/18)) := by
  sorry

#check train_passing_time

end train_passing_time_l2527_252735


namespace x_squared_plus_reciprocal_l2527_252792

theorem x_squared_plus_reciprocal (x : ℝ) (h : 59 = x^4 + 1/x^4) : x^2 + 1/x^2 = Real.sqrt 61 := by
  sorry

end x_squared_plus_reciprocal_l2527_252792


namespace mean_temperature_is_zero_l2527_252708

def temperatures : List ℤ := [-3, -1, -6, 0, 4, 6]

theorem mean_temperature_is_zero : 
  (temperatures.sum : ℚ) / temperatures.length = 0 := by
  sorry

end mean_temperature_is_zero_l2527_252708


namespace specific_tetrahedron_volume_l2527_252782

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 3 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 3,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := Real.sqrt 34,
    RS := Real.sqrt 41
  }
  volume t = 3 := by sorry

end specific_tetrahedron_volume_l2527_252782


namespace derek_journey_l2527_252702

/-- Proves that given a journey where half the distance is traveled at 20 km/h 
    and the other half at 4 km/h, with a total travel time of 54 minutes, 
    the distance walked is 3.0 km. -/
theorem derek_journey (total_distance : ℝ) (total_time : ℝ) : 
  (total_distance / 2) / 20 + (total_distance / 2) / 4 = total_time ∧
  total_time = 54 / 60 →
  total_distance / 2 = 3 :=
by sorry

end derek_journey_l2527_252702


namespace bowtie_equation_solution_l2527_252713

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  3 * a + Real.sqrt (4 * b + Real.sqrt (4 * b + Real.sqrt (4 * b + Real.sqrt (4 * b))))

-- Theorem statement
theorem bowtie_equation_solution (y : ℝ) : bowtie 5 y = 20 → y = 5 := by
  sorry

end bowtie_equation_solution_l2527_252713


namespace point_on_line_ratio_l2527_252704

/-- Given five points O, A, B, C, D on a straight line with specified distances,
    and a point P between B and C satisfying a ratio condition,
    prove that OP has the given value. -/
theorem point_on_line_ratio (a b c d k : ℝ) :
  let OA := a
  let OB := k * b
  let OC := c
  let OD := k * d
  ∀ P : ℝ, OB ≤ P ∧ P ≤ OC →
  (a - P) / (P - k * d) = k * (k * b - P) / (P - c) →
  P = (a * c + k * b * d) / (a + c - k * b + k * d - 1 + k) :=
by sorry

end point_on_line_ratio_l2527_252704


namespace school_pet_ownership_stats_l2527_252744

/-- Represents the school statistics -/
structure SchoolStats where
  total_students : ℕ
  cat_owners : ℕ
  dog_owners : ℕ

/-- Calculates the percentage of students owning a specific pet -/
def pet_ownership_percentage (stats : SchoolStats) (pet_owners : ℕ) : ℚ :=
  (pet_owners : ℚ) / (stats.total_students : ℚ) * 100

/-- Calculates the percent difference between two percentages -/
def percent_difference (p1 p2 : ℚ) : ℚ :=
  abs (p1 - p2)

/-- Theorem stating the correctness of the calculated percentages -/
theorem school_pet_ownership_stats (stats : SchoolStats) 
  (h1 : stats.total_students = 500)
  (h2 : stats.cat_owners = 80)
  (h3 : stats.dog_owners = 100) :
  pet_ownership_percentage stats stats.cat_owners = 16 ∧
  percent_difference (pet_ownership_percentage stats stats.dog_owners) (pet_ownership_percentage stats stats.cat_owners) = 4 := by
  sorry

#eval pet_ownership_percentage ⟨500, 80, 100⟩ 80
#eval percent_difference (pet_ownership_percentage ⟨500, 80, 100⟩ 100) (pet_ownership_percentage ⟨500, 80, 100⟩ 80)

end school_pet_ownership_stats_l2527_252744


namespace candy_pebbles_l2527_252781

theorem candy_pebbles (candy : ℕ) (lance : ℕ) : 
  lance = 3 * candy ∧ lance = candy + 8 → candy = 4 :=
by sorry

end candy_pebbles_l2527_252781


namespace no_periodic_difference_with_3_and_pi_periods_l2527_252796

-- Define a periodic function
def isPeriodic (f : ℝ → ℝ) :=
  (∃ x y, f x ≠ f y) ∧ (∃ p > 0, ∀ x, f (x + p) = f x)

-- Define the period of a function
def isPeriodOf (p : ℝ) (f : ℝ → ℝ) :=
  p > 0 ∧ ∀ x, f (x + p) = f x

-- Theorem statement
theorem no_periodic_difference_with_3_and_pi_periods :
  ¬ ∃ (g h : ℝ → ℝ),
    isPeriodic g ∧ isPeriodic h ∧
    isPeriodOf 3 g ∧ isPeriodOf π h ∧
    isPeriodic (g - h) :=
sorry

end no_periodic_difference_with_3_and_pi_periods_l2527_252796


namespace solve_system_l2527_252756

theorem solve_system (x y : ℤ) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := by
  sorry

end solve_system_l2527_252756


namespace compound_propositions_truth_count_l2527_252797

theorem compound_propositions_truth_count
  (p q : Prop)
  (hp : p)
  (hq : ¬q) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) ∧ (¬q) :=
by sorry

end compound_propositions_truth_count_l2527_252797


namespace senior_mean_score_l2527_252778

theorem senior_mean_score (total_students : ℕ) (overall_mean : ℚ) 
  (senior_count : ℕ) (non_senior_count : ℕ) (senior_mean : ℚ) (non_senior_mean : ℚ) :
  total_students = 120 →
  overall_mean = 110 →
  non_senior_count = 2 * senior_count →
  senior_mean = (3/2) * non_senior_mean →
  senior_count + non_senior_count = total_students →
  (senior_count * senior_mean + non_senior_count * non_senior_mean) / total_students = overall_mean →
  senior_mean = 141.43 := by
sorry

#eval (141.43 : ℚ)

end senior_mean_score_l2527_252778


namespace thirty_percent_less_than_ninety_l2527_252774

theorem thirty_percent_less_than_ninety (x : ℝ) : x + (1/4) * x = 90 - 0.3 * 90 → x = 50 := by
  sorry

end thirty_percent_less_than_ninety_l2527_252774


namespace water_container_problem_l2527_252752

theorem water_container_problem :
  let large_capacity : ℚ := 144
  let small_capacity : ℚ := 100
  ∀ x y : ℚ,
  (x + (4/5) * y = large_capacity) →
  (y + (5/12) * x = small_capacity) →
  x = 96 ∧ y = 60 := by
  sorry

end water_container_problem_l2527_252752


namespace potions_needed_for_owl_l2527_252791

/-- The number of Knuts in a Sickle -/
def knuts_per_sickle : ℕ := 23

/-- The number of Sickles in a Galleon -/
def sickles_per_galleon : ℕ := 17

/-- The cost of the owl in Galleons, Sickles, and Knuts -/
def owl_cost : ℕ × ℕ × ℕ := (2, 1, 5)

/-- The worth of each potion in Knuts -/
def potion_worth : ℕ := 9

/-- The function to calculate the total cost in Knuts -/
def total_cost_in_knuts (cost : ℕ × ℕ × ℕ) : ℕ :=
  cost.1 * sickles_per_galleon * knuts_per_sickle + 
  cost.2.1 * knuts_per_sickle + 
  cost.2.2

/-- The theorem stating the number of potions needed -/
theorem potions_needed_for_owl : 
  (total_cost_in_knuts owl_cost) / potion_worth = 90 := by
  sorry

end potions_needed_for_owl_l2527_252791


namespace equation_solution_l2527_252753

theorem equation_solution : ∃ y : ℝ, (32 : ℝ) ^ (3 * y) = 8 ^ (2 * y + 1) ∧ y = 1/3 := by
  sorry

end equation_solution_l2527_252753


namespace m_range_l2527_252727

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := 2 < m ∧ m < 4

def q (m : ℝ) : Prop := m < 0 ∨ m > 3

-- Define the range of m
def range_m (m : ℝ) : Prop := m < 0 ∨ (2 < m ∧ m < 3) ∨ m > 3

-- State the theorem
theorem m_range : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ range_m m :=
by sorry

end m_range_l2527_252727


namespace salary_after_changes_l2527_252710

-- Define the initial salary
def initial_salary : ℝ := 2000

-- Define the raise percentage
def raise_percentage : ℝ := 0.20

-- Define the pay cut percentage
def pay_cut_percentage : ℝ := 0.20

-- Theorem to prove
theorem salary_after_changes (s : ℝ) (r : ℝ) (c : ℝ) 
  (h1 : s = initial_salary) 
  (h2 : r = raise_percentage) 
  (h3 : c = pay_cut_percentage) : 
  s * (1 + r) * (1 - c) = 1920 := by
  sorry

end salary_after_changes_l2527_252710


namespace total_players_on_ground_l2527_252758

/-- The number of cricket players -/
def cricket_players : ℕ := 15

/-- The number of hockey players -/
def hockey_players : ℕ := 12

/-- The number of football players -/
def football_players : ℕ := 13

/-- The number of softball players -/
def softball_players : ℕ := 15

/-- Theorem stating the total number of players on the ground -/
theorem total_players_on_ground :
  cricket_players + hockey_players + football_players + softball_players = 55 := by
  sorry

end total_players_on_ground_l2527_252758


namespace part_one_part_two_l2527_252741

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 2*x^2 - 3*x + 1 ≤ 0}
def Q (a : ℝ) : Set ℝ := {x : ℝ | (x - a)*(x - a - 1) ≤ 0}

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Part 1: Prove that when a = 1, (∁_U P) ∩ Q = {x | 1 < x ≤ 2}
theorem part_one : (Set.compl P) ∩ (Q 1) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

-- Part 2: Prove that P ∩ Q = P if and only if a ∈ [0, 1/2]
theorem part_two : ∀ a : ℝ, P ∩ (Q a) = P ↔ 0 ≤ a ∧ a ≤ 1/2 := by sorry

end part_one_part_two_l2527_252741


namespace total_tulips_l2527_252766

def arwen_tulips : ℕ := 20

def elrond_tulips (a : ℕ) : ℕ := 2 * a

theorem total_tulips : arwen_tulips + elrond_tulips arwen_tulips = 60 := by
  sorry

end total_tulips_l2527_252766


namespace complex_number_in_fourth_quadrant_l2527_252785

/-- The complex number z defined as (i+2)/i is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (Complex.I + 2) / Complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l2527_252785


namespace dinner_slices_l2527_252764

/-- Represents the number of slices of pie served during different times of the day -/
structure PieSlices where
  lunch : ℕ
  total : ℕ

/-- Proves that the number of slices served during dinner is 5,
    given 7 slices were served during lunch and 12 slices in total -/
theorem dinner_slices (ps : PieSlices) 
  (h_lunch : ps.lunch = 7) 
  (h_total : ps.total = 12) : 
  ps.total - ps.lunch = 5 := by
  sorry

end dinner_slices_l2527_252764


namespace mean_temperature_l2527_252707

def temperatures : List ℤ := [-8, -5, -3, 0, 4, 2, 7]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -3/7 := by sorry

end mean_temperature_l2527_252707


namespace fraction_multiplication_addition_l2527_252709

theorem fraction_multiplication_addition : (1 / 3 : ℚ) * (3 / 4 : ℚ) * (1 / 5 : ℚ) + (1 / 6 : ℚ) = 13 / 60 := by
  sorry

end fraction_multiplication_addition_l2527_252709


namespace other_number_is_nine_l2527_252798

theorem other_number_is_nine (x : ℝ) (h1 : (x + 5) / 2 = 7) : x = 9 := by
  sorry

end other_number_is_nine_l2527_252798


namespace quadratic_equation_proof_l2527_252772

theorem quadratic_equation_proof (x₁ x₂ k : ℝ) : 
  (x₁^2 - 6*x₁ + k = 0) →
  (x₂^2 - 6*x₂ + k = 0) →
  (x₁^2 * x₂^2 - x₁ - x₂ = 115) →
  (k = -11 ∧ ((x₁ = 3 + 2*Real.sqrt 5 ∧ x₂ = 3 - 2*Real.sqrt 5) ∨ 
              (x₁ = 3 - 2*Real.sqrt 5 ∧ x₂ = 3 + 2*Real.sqrt 5))) :=
by sorry

end quadratic_equation_proof_l2527_252772


namespace system_coefficients_proof_l2527_252722

theorem system_coefficients_proof : ∃! (a b c : ℝ),
  (∀ x y : ℝ, a * (x - 1) + 2 * y ≠ 1 ∨ b * (x - 1) + c * y ≠ 3) ∧
  (a * (-1/4) + 2 * (5/8) = 1) ∧
  (b * (1/4) + c * (5/8) = 3) ∧
  a = 1 ∧ b = 2 ∧ c = 4 := by
  sorry

end system_coefficients_proof_l2527_252722


namespace rocky_ran_36_miles_l2527_252743

/-- Rocky's training schedule for the first three days -/
def rocky_training : ℕ → ℕ
| 1 => 4  -- Day one: 4 miles
| 2 => 2 * rocky_training 1  -- Day two: double of day one
| 3 => 3 * rocky_training 2  -- Day three: triple of day two
| _ => 0  -- Other days (not relevant for this problem)

/-- The total miles Rocky ran in the first three days of training -/
def total_miles : ℕ := rocky_training 1 + rocky_training 2 + rocky_training 3

/-- Theorem stating that Rocky ran 36 miles in total during the first three days of training -/
theorem rocky_ran_36_miles : total_miles = 36 := by
  sorry

end rocky_ran_36_miles_l2527_252743


namespace min_value_x_plus_4y_l2527_252700

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/(2*y) = 1) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 1/(2*w) = 1 → x + 4*y ≤ z + 4*w ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = 1 ∧ a + 4*b = 3 + 2 * Real.sqrt 2 := by
sorry

end min_value_x_plus_4y_l2527_252700


namespace smallest_integer_with_divisibility_condition_l2527_252729

def is_divisible (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

theorem smallest_integer_with_divisibility_condition :
  ∃ (n : ℕ) (i j : ℕ),
    n > 0 ∧
    i < j ∧
    j - i = 1 ∧
    j ≤ 30 ∧
    (∀ k : ℕ, k ≤ 30 → k ≠ i → k ≠ j → is_divisible n k) ∧
    ¬(is_divisible n i) ∧
    ¬(is_divisible n j) ∧
    (∀ m : ℕ, m > 0 →
      (∃ (x y : ℕ), x < y ∧ y - x = 1 ∧ y ≤ 30 ∧
        (∀ k : ℕ, k ≤ 30 → k ≠ x → k ≠ y → is_divisible m k) ∧
        ¬(is_divisible m x) ∧
        ¬(is_divisible m y)) →
      m ≥ n) ∧
    n = 2230928700 :=
by sorry

end smallest_integer_with_divisibility_condition_l2527_252729


namespace S_min_value_l2527_252731

/-- The function S defined on real numbers x and y -/
def S (x y : ℝ) : ℝ := 2 * x^2 - x*y + y^2 + 2*x + 3*y

/-- Theorem stating that S has a minimum value of -4 -/
theorem S_min_value :
  (∀ x y : ℝ, S x y ≥ -4) ∧ (∃ x y : ℝ, S x y = -4) :=
sorry

end S_min_value_l2527_252731


namespace bus_problem_l2527_252768

/-- Proof of the number of people who got off at the first bus stop -/
theorem bus_problem (total_rows : Nat) (seats_per_row : Nat) 
  (initial_boarding : Nat) (first_stop_boarding : Nat) 
  (second_stop_boarding : Nat) (second_stop_departing : Nat) 
  (empty_seats_after_second : Nat) : 
  total_rows = 23 → 
  seats_per_row = 4 → 
  initial_boarding = 16 → 
  first_stop_boarding = 15 → 
  second_stop_boarding = 17 → 
  second_stop_departing = 10 → 
  empty_seats_after_second = 57 → 
  ∃ (first_stop_departing : Nat), 
    first_stop_departing = 3 ∧
    (total_rows * seats_per_row) - 
    (initial_boarding + first_stop_boarding + second_stop_boarding - 
     first_stop_departing - second_stop_departing) = 
    empty_seats_after_second :=
by sorry

end bus_problem_l2527_252768


namespace range_of_c_l2527_252799

/-- The range of c for which y = c^x is a decreasing function and x^2 - √2x + c > 0 does not hold for all x ∈ ℝ -/
theorem range_of_c (c : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → c^x₁ > c^x₂) → -- y = c^x is a decreasing function
  (¬∀ x : ℝ, x^2 - Real.sqrt 2 * x + c > 0) → -- negation of q
  ((∀ x₁ x₂ : ℝ, x₁ < x₂ → c^x₁ > c^x₂) ∨ (∀ x : ℝ, x^2 - Real.sqrt 2 * x + c > 0)) → -- p or q
  0 < c ∧ c ≤ (1/2 : ℝ) :=
by sorry

end range_of_c_l2527_252799


namespace simplify_and_evaluate_l2527_252745

theorem simplify_and_evaluate (y : ℝ) :
  let x : ℝ := -4
  ((x + y)^2 - y * (2 * x + y) - 8 * x) / (2 * x) = -6 := by
  sorry

end simplify_and_evaluate_l2527_252745


namespace inverse_square_direct_cube_relation_l2527_252712

/-- Given that x varies inversely as the square of y and directly as the cube of z,
    prove that when x = 1 for y = 3 and z = 2, then x = 8/9 when y = 9 and z = 4. -/
theorem inverse_square_direct_cube_relation (k : ℚ) :
  (1 : ℚ) = k * (2^3 : ℚ) / (3^2 : ℚ) →
  (8/9 : ℚ) = k * (4^3 : ℚ) / (9^2 : ℚ) :=
by sorry

end inverse_square_direct_cube_relation_l2527_252712


namespace m_is_always_odd_l2527_252742

theorem m_is_always_odd (a b : ℤ) (h1 : b = a + 1) (c : ℤ) (h2 : c = a * b) :
  ∃ (M : ℤ), M^2 = a^2 + b^2 + c^2 ∧ Odd M := by
  sorry

end m_is_always_odd_l2527_252742


namespace danny_soda_distribution_l2527_252716

theorem danny_soda_distribution (initial_bottles : ℝ) (drunk_percentage : ℝ) (remaining_percentage : ℝ) : 
  initial_bottles = 3 →
  drunk_percentage = 90 →
  remaining_percentage = 70 →
  let drunk_amount := (drunk_percentage / 100) * 1
  let remaining_amount := (remaining_percentage / 100) * 1
  let given_away := initial_bottles - (drunk_amount + remaining_amount)
  given_away / 2 = 0.7 := by
  sorry

end danny_soda_distribution_l2527_252716


namespace radii_ratio_in_regular_hexagonal_pyramid_l2527_252787

/-- A regular hexagonal pyramid with circumscribed and inscribed spheres -/
structure RegularHexagonalPyramid where
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- The center of the circumscribed sphere lies on the surface of the inscribed sphere -/
  center_on_surface : R = r * (1 + Real.sqrt 21 / 3)

/-- The ratio of the radii of the circumscribed sphere to the inscribed sphere
    in a regular hexagonal pyramid where the center of the circumscribed sphere
    lies on the surface of the inscribed sphere is (3 + √21) / 3 -/
theorem radii_ratio_in_regular_hexagonal_pyramid (p : RegularHexagonalPyramid) :
  p.R / p.r = (3 + Real.sqrt 21) / 3 := by
  sorry

end radii_ratio_in_regular_hexagonal_pyramid_l2527_252787


namespace total_timeout_time_total_timeout_is_185_minutes_l2527_252714

/-- Calculates the total time students spend in time-out given the number of time-outs for different offenses and the duration of each time-out. -/
theorem total_timeout_time (running_timeouts : ℕ) (timeout_duration : ℕ) : ℕ :=
  let food_timeouts := 5 * running_timeouts - 1
  let swearing_timeouts := food_timeouts / 3
  let total_timeouts := running_timeouts + food_timeouts + swearing_timeouts
  total_timeouts * timeout_duration

/-- Proves that the total time students spend in time-out is 185 minutes under the given conditions. -/
theorem total_timeout_is_185_minutes : total_timeout_time 5 5 = 185 := by
  sorry

end total_timeout_time_total_timeout_is_185_minutes_l2527_252714


namespace three_person_subcommittees_l2527_252724

theorem three_person_subcommittees (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by
  sorry

end three_person_subcommittees_l2527_252724


namespace no_valid_polygon_pairs_l2527_252763

theorem no_valid_polygon_pairs : ¬∃ (r k : ℕ), 
  r > 2 ∧ k > 2 ∧ 
  (180 * r - 360) / (180 * k - 360) = 7 / 5 ∧
  ∃ (c : ℚ), c * r = k := by
  sorry

end no_valid_polygon_pairs_l2527_252763


namespace jinas_koala_bears_l2527_252717

theorem jinas_koala_bears :
  let initial_teddies : ℕ := 5
  let bunny_multiplier : ℕ := 3
  let additional_teddies_per_bunny : ℕ := 2
  let total_mascots : ℕ := 51
  let bunnies : ℕ := initial_teddies * bunny_multiplier
  let additional_teddies : ℕ := bunnies * additional_teddies_per_bunny
  let total_teddies : ℕ := initial_teddies + additional_teddies
  let koala_bears : ℕ := total_mascots - (total_teddies + bunnies)
  koala_bears = 1 := by
  sorry

end jinas_koala_bears_l2527_252717


namespace max_value_complex_expression_l2527_252754

theorem max_value_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  Complex.abs ((w - 2)^2 * (w + 2)) ≤ 24 ∧
  ∃ w₀ : ℂ, Complex.abs w₀ = 2 ∧ Complex.abs ((w₀ - 2)^2 * (w₀ + 2)) = 24 :=
by sorry

end max_value_complex_expression_l2527_252754


namespace apple_distribution_problem_l2527_252726

/-- The number of ways to distribute n indistinguishable objects among k distinguishable boxes --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute apples among people --/
def apple_distribution (total_apples min_apples people : ℕ) : ℕ :=
  stars_and_bars (total_apples - people * min_apples) people

theorem apple_distribution_problem : apple_distribution 30 3 3 = 253 := by
  sorry

end apple_distribution_problem_l2527_252726


namespace two_questions_sufficient_l2527_252773

/-- Represents a person who is either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- Represents a position on a 2D plane -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Represents the table with 10 people -/
structure Table :=
  (people : Fin 10 → Person)
  (positions : Fin 10 → Position)

/-- A function that simulates asking a question about the distance to the nearest liar -/
def askQuestion (t : Table) (travelerPos : Position) : Fin 10 → ℝ :=
  sorry

/-- The main theorem stating that 2 questions are sufficient to identify all liars -/
theorem two_questions_sufficient (t : Table) :
  ∃ (pos1 pos2 : Position),
    (∀ (p : Fin 10), t.people p = Person.Liar ↔
      ∃ (q : Fin 10), t.people q = Person.Liar ∧
        askQuestion t pos1 p ≠ askQuestion t pos1 q ∨
        askQuestion t pos2 p ≠ askQuestion t pos2 q) :=
sorry

end two_questions_sufficient_l2527_252773


namespace exists_divisible_by_15_with_sqrt_between_30_and_30_5_l2527_252795

theorem exists_divisible_by_15_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, 15 ∣ n ∧ 30 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 30.5 :=
by
  use 900
  sorry

end exists_divisible_by_15_with_sqrt_between_30_and_30_5_l2527_252795


namespace brothers_age_ratio_l2527_252711

/-- Represents the ages of three brothers: Richard, David, and Scott -/
structure BrothersAges where
  david : ℕ
  richard : ℕ
  scott : ℕ

/-- Calculates the ages of the brothers after a given number of years -/
def agesAfterYears (ages : BrothersAges) (years : ℕ) : BrothersAges :=
  { david := ages.david + years
  , richard := ages.richard + years
  , scott := ages.scott + years }

/-- The theorem statement based on the given problem -/
theorem brothers_age_ratio : ∀ (ages : BrothersAges),
  ages.richard = ages.david + 6 →
  ages.david = ages.scott + 8 →
  ages.david = 14 →
  ∃ (k : ℕ), (agesAfterYears ages 8).richard = k * (agesAfterYears ages 8).scott →
  (agesAfterYears ages 8).richard / (agesAfterYears ages 8).scott = 2 := by
  sorry

end brothers_age_ratio_l2527_252711


namespace max_total_points_l2527_252771

/-- Represents the carnival game setup and Tiffany's current state -/
structure CarnivalGame where
  initial_money : ℕ := 3
  game_cost : ℕ := 1
  rings_per_game : ℕ := 5
  red_points : ℕ := 2
  green_points : ℕ := 3
  blue_points : ℕ := 5
  blue_success_rate : ℚ := 1/10
  time_limit : ℕ := 1
  games_played : ℕ := 2
  current_red : ℕ := 4
  current_green : ℕ := 5
  current_blue : ℕ := 1

/-- Calculates the maximum possible points for a single game -/
def max_points_per_game (game : CarnivalGame) : ℕ :=
  game.rings_per_game * game.blue_points

/-- Calculates the current total points -/
def current_total_points (game : CarnivalGame) : ℕ :=
  game.current_red * game.red_points +
  game.current_green * game.green_points +
  game.current_blue * game.blue_points

/-- Theorem: The maximum total points Tiffany can achieve in three games is 53 -/
theorem max_total_points (game : CarnivalGame) :
  current_total_points game + max_points_per_game game = 53 :=
sorry

end max_total_points_l2527_252771


namespace rectangular_field_diagonal_shortcut_l2527_252719

theorem rectangular_field_diagonal_shortcut (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt (x^2 + y^2) + x/2 = x + y) → (min x y)/(max x y) = 3/4 := by
  sorry

end rectangular_field_diagonal_shortcut_l2527_252719


namespace cos_negative_1320_degrees_l2527_252759

theorem cos_negative_1320_degrees : Real.cos (-(1320 * π / 180)) = -1/2 := by
  sorry

end cos_negative_1320_degrees_l2527_252759


namespace system_solution_l2527_252783

theorem system_solution (x y : ℝ) :
  (2 / (x^2 + y^2) + x^2 * y^2 = 2) ∧
  (x^4 + y^4 + 3 * x^2 * y^2 = 5) ↔
  ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end system_solution_l2527_252783


namespace parallel_vectors_cos_relation_l2527_252775

/-- Given two parallel vectors a and b, prove that cos(π/2 + α) = -1/3 -/
theorem parallel_vectors_cos_relation (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h1 : a = (1/3, Real.tan α))
  (h2 : b = (Real.cos α, 1))
  (h3 : ∃ (k : ℝ), a = k • b) : 
  Real.cos (π/2 + α) = -1/3 := by
  sorry

end parallel_vectors_cos_relation_l2527_252775


namespace sin_alpha_plus_pi_l2527_252738

theorem sin_alpha_plus_pi (α : Real) :
  (∃ P : ℝ × ℝ, P.1 = Real.sin (5 * Real.pi / 3) ∧ P.2 = Real.cos (5 * Real.pi / 3) ∧
   P.1 = Real.sin α ∧ P.2 = Real.cos α) →
  Real.sin (α + Real.pi) = -1/2 := by
sorry

end sin_alpha_plus_pi_l2527_252738


namespace negation_of_existence_negation_of_specific_proposition_l2527_252760

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ p x) ↔ (∀ x : ℝ, x ≥ 0 → ¬ p x) := by
  sorry

-- The specific proposition
def proposition (x : ℝ) : Prop := 2 * x = 3

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ proposition x) ↔ (∀ x : ℝ, x ≥ 0 → 2 * x ≠ 3) := by
  sorry

end negation_of_existence_negation_of_specific_proposition_l2527_252760


namespace borrowed_sum_proof_l2527_252770

/-- 
Given a principal P borrowed at 8% per annum simple interest for 8 years,
if the interest I is equal to P - 900, then P = 2500.
-/
theorem borrowed_sum_proof (P : ℝ) (I : ℝ) : 
  (I = P * 8 * 8 / 100) →   -- Simple interest formula
  (I = P - 900) →           -- Given condition
  P = 2500 := by
sorry

end borrowed_sum_proof_l2527_252770


namespace line_through_parabola_focus_l2527_252730

/-- The focus of a parabola y² = 4x is the point (1, 0) -/
def focus_of_parabola : ℝ × ℝ := (1, 0)

/-- A line passing through a point (x, y) is represented by the equation ax - y + 1 = 0 -/
def line_passes_through (a : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 - p.2 + 1 = 0

theorem line_through_parabola_focus (a : ℝ) :
  line_passes_through a focus_of_parabola → a = -1 := by
  sorry

end line_through_parabola_focus_l2527_252730


namespace complex_product_example_l2527_252748

-- Define the complex numbers
def z₁ : ℂ := 3 + 4 * Complex.I
def z₂ : ℂ := -2 - 3 * Complex.I

-- State the theorem
theorem complex_product_example : z₁ * z₂ = -18 - 17 * Complex.I := by
  sorry

end complex_product_example_l2527_252748


namespace sin_minus_cos_eq_one_l2527_252718

theorem sin_minus_cos_eq_one (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  (Real.sin x - Real.cos x = 1 ↔ x = Real.pi / 2 ∨ x = Real.pi) := by
sorry

end sin_minus_cos_eq_one_l2527_252718


namespace custom_mult_value_l2527_252739

/-- Custom multiplication operation for non-zero integers -/
def custom_mult (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem custom_mult_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a + b = 12 → a * b = 32 → custom_mult a b = 3 / 8 := by
  sorry

end custom_mult_value_l2527_252739


namespace count_numbers_with_at_most_two_digits_is_2151_l2527_252776

/-- The count of positive integers less than 100,000 with at most two different digits -/
def count_numbers_with_at_most_two_digits : ℕ :=
  let max_number := 100000
  let single_digit_count := 9 * 5
  let two_digits_without_zero := 36 * (2^2 - 2 + 2^3 - 2 + 2^4 - 2 + 2^5 - 2)
  let two_digits_with_zero := 9 * (2^1 - 1 + 2^2 - 1 + 2^3 - 1 + 2^4 - 1)
  single_digit_count + two_digits_without_zero + two_digits_with_zero

theorem count_numbers_with_at_most_two_digits_is_2151 :
  count_numbers_with_at_most_two_digits = 2151 :=
by sorry

end count_numbers_with_at_most_two_digits_is_2151_l2527_252776


namespace expression_equality_l2527_252705

theorem expression_equality : 2 + 2/3 + 6.3 - (5/3 - (1 + 3/5)) = 8.9 := by
  sorry

end expression_equality_l2527_252705


namespace m_plus_n_value_l2527_252725

theorem m_plus_n_value (m n : ℚ) :
  (∀ x, x^2 + m*x + 6 = (x-2)*(x-n)) →
  m + n = -2 := by
sorry

end m_plus_n_value_l2527_252725


namespace exists_satisfying_quadratic_l2527_252762

/-- A quadratic function satisfying the given conditions -/
def satisfying_quadratic (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  (∀ x, |x| ≤ 1 → |f x| ≤ 1) ∧
  (|f 2| ≥ 7)

/-- There exists a quadratic function satisfying the given conditions -/
theorem exists_satisfying_quadratic : ∃ f : ℝ → ℝ, satisfying_quadratic f := by
  sorry

end exists_satisfying_quadratic_l2527_252762


namespace root_equation_value_l2527_252765

theorem root_equation_value (m : ℝ) (h : m^2 - 3*m - 1 = 0) : 2*m^2 - 6*m + 5 = 7 := by
  sorry

end root_equation_value_l2527_252765


namespace sqrt_eight_plus_sqrt_two_l2527_252734

theorem sqrt_eight_plus_sqrt_two : Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 := by
  sorry

end sqrt_eight_plus_sqrt_two_l2527_252734


namespace benny_turnips_l2527_252794

theorem benny_turnips (melanie_turnips benny_turnips total_turnips : ℕ) 
  (h1 : melanie_turnips = 139)
  (h2 : total_turnips = 252)
  (h3 : melanie_turnips + benny_turnips = total_turnips) : 
  benny_turnips = 113 := by
  sorry

end benny_turnips_l2527_252794


namespace function_always_positive_implies_a_less_than_one_l2527_252723

theorem function_always_positive_implies_a_less_than_one :
  (∀ x : ℝ, |x - 1| + |x - 2| - a > 0) → a < 1 := by
  sorry

end function_always_positive_implies_a_less_than_one_l2527_252723


namespace flowerbed_fraction_is_one_eighth_l2527_252703

/-- Represents a rectangular park with flower beds -/
structure Park where
  /-- Length of the shorter parallel side of the trapezoidal area -/
  short_side : ℝ
  /-- Length of the longer parallel side of the trapezoidal area -/
  long_side : ℝ
  /-- Number of congruent isosceles right triangle flower beds -/
  num_flowerbeds : ℕ

/-- The fraction of the park occupied by flower beds -/
def flowerbed_fraction (p : Park) : ℝ :=
  -- Define the fraction calculation here
  sorry

/-- Theorem stating that for a park with specific dimensions, 
    the fraction of area occupied by flower beds is 1/8 -/
theorem flowerbed_fraction_is_one_eighth :
  ∀ (p : Park), 
  p.short_side = 30 ∧ 
  p.long_side = 50 ∧ 
  p.num_flowerbeds = 3 →
  flowerbed_fraction p = 1/8 :=
by
  sorry

end flowerbed_fraction_is_one_eighth_l2527_252703


namespace cathy_final_state_l2527_252790

/-- Represents a direction on the coordinate plane -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a position on the coordinate plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents Cathy's state after each move -/
structure CathyState :=
  (position : Position)
  (direction : Direction)
  (moveNumber : Nat)
  (distanceTraveled : Nat)

/-- Calculates the next direction after turning right -/
def turnRight (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

/-- Calculates the distance for a given move number -/
def moveDistance (n : Nat) : Nat :=
  2 * n

/-- Updates the position based on the current direction and distance -/
def updatePosition (p : Position) (d : Direction) (dist : Nat) : Position :=
  match d with
  | Direction.North => ⟨p.x, p.y + dist⟩
  | Direction.East => ⟨p.x + dist, p.y⟩
  | Direction.South => ⟨p.x, p.y - dist⟩
  | Direction.West => ⟨p.x - dist, p.y⟩

/-- Performs a single move and updates Cathy's state -/
def move (state : CathyState) : CathyState :=
  let newMoveNumber := state.moveNumber + 1
  let distance := moveDistance newMoveNumber
  let newPosition := updatePosition state.position state.direction distance
  let newDirection := turnRight state.direction
  let newDistanceTraveled := state.distanceTraveled + distance
  ⟨newPosition, newDirection, newMoveNumber, newDistanceTraveled⟩

/-- Performs n moves starting from the given initial state -/
def performMoves (initialState : CathyState) (n : Nat) : CathyState :=
  match n with
  | 0 => initialState
  | m + 1 => move (performMoves initialState m)

/-- The main theorem to prove -/
theorem cathy_final_state :
  let initialState : CathyState := ⟨⟨2, -3⟩, Direction.North, 0, 0⟩
  let finalState := performMoves initialState 12
  finalState.position = ⟨-10, -15⟩ ∧ finalState.distanceTraveled = 146 := by
  sorry


end cathy_final_state_l2527_252790
