import Mathlib

namespace expression_bounds_l865_86549

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1/2) (hd : 0 ≤ d ∧ d ≤ 1/2) :
  let expr := Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
               Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2)
  2 * Real.sqrt 2 ≤ expr ∧ expr ≤ 4 ∧ 
  ∀ x, 2 * Real.sqrt 2 ≤ x ∧ x ≤ 4 → ∃ a b c d, 
    0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1/2 ∧ 0 ≤ d ∧ d ≤ 1/2 ∧
    x = Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
        Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2) := by
  sorry

end expression_bounds_l865_86549


namespace complement_intersection_theorem_l865_86505

def U : Set ℕ := {2, 3, 5, 7, 8}
def A : Set ℕ := {2, 8}
def B : Set ℕ := {3, 5, 8}

theorem complement_intersection_theorem : (U \ A) ∩ B = {3, 5} := by sorry

end complement_intersection_theorem_l865_86505


namespace problem_solution_l865_86571

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

theorem problem_solution (a : ℝ) :
  (f' a 3 = 0 → a = 3) ∧
  (∀ x < 0, Monotone (f a) ↔ a ≥ 0) := by sorry

end problem_solution_l865_86571


namespace triangle_area_specific_l865_86531

/-- The area of a triangle with two sides of length 31 and one side of length 40 is 474 -/
theorem triangle_area_specific : ∃ (A : ℝ), 
  A = (Real.sqrt (51 * (51 - 31) * (51 - 31) * (51 - 40)) : ℝ) ∧ A = 474 := by
  sorry

end triangle_area_specific_l865_86531


namespace frisbee_deck_difference_l865_86555

/-- Represents the number of items Bella has -/
structure BellasItems where
  marbles : ℕ
  frisbees : ℕ
  deckCards : ℕ

/-- The conditions of the problem -/
def problemConditions (items : BellasItems) : Prop :=
  items.marbles = 2 * items.frisbees ∧
  items.marbles = 60 ∧
  (items.marbles + 2/5 * items.marbles + 
   items.frisbees + 2/5 * items.frisbees + 
   items.deckCards + 2/5 * items.deckCards) = 140

/-- The theorem to prove -/
theorem frisbee_deck_difference (items : BellasItems) 
  (h : problemConditions items) : 
  items.frisbees - items.deckCards = 20 := by
  sorry


end frisbee_deck_difference_l865_86555


namespace arithmetic_progression_sum_110_l865_86594

/-- Given an arithmetic progression with first term a and common difference d -/
def arithmetic_progression (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1) * d

/-- Sum of first n terms of an arithmetic progression -/
def sum_arithmetic_progression (a d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_progression_sum_110 
  (a d : ℚ) 
  (h1 : sum_arithmetic_progression a d 10 = 100)
  (h2 : sum_arithmetic_progression a d 100 = 10) :
  sum_arithmetic_progression a d 110 = -110 := by
  sorry

end arithmetic_progression_sum_110_l865_86594


namespace scientific_notation_of_20_8_billion_l865_86525

/-- Expresses 20.8 billion in scientific notation -/
theorem scientific_notation_of_20_8_billion :
  20.8 * (10 : ℝ)^9 = 2.08 * (10 : ℝ)^9 := by sorry

end scientific_notation_of_20_8_billion_l865_86525


namespace count_valid_domains_l865_86573

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the set of possible domain elements
def domain_elements : Set ℝ := {-Real.sqrt 2, -1, 1, Real.sqrt 2}

-- Define the range
def target_range : Set ℝ := {1, 2}

-- Define a valid domain
def is_valid_domain (S : Set ℝ) : Prop :=
  S ⊆ domain_elements ∧ f '' S = target_range

-- Theorem statement
theorem count_valid_domains :
  ∃ (valid_domains : Finset (Set ℝ)),
    (∀ S ∈ valid_domains, is_valid_domain S) ∧
    (∀ S, is_valid_domain S → S ∈ valid_domains) ∧
    valid_domains.card = 9 := by
  sorry

end count_valid_domains_l865_86573


namespace average_sale_per_month_l865_86513

def sales : List ℕ := [120, 80, 100, 140, 160]

theorem average_sale_per_month : 
  (List.sum sales) / (List.length sales) = 120 := by
  sorry

end average_sale_per_month_l865_86513


namespace max_value_sqrt_sum_l865_86572

theorem max_value_sqrt_sum (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_two : x + y + z = 2) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ 
    Real.sqrt a + Real.sqrt (2 * b) + Real.sqrt (3 * c) > Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z)) 
  ∨ 
  Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z) = 2 * Real.sqrt 3 :=
sorry

end max_value_sqrt_sum_l865_86572


namespace complex_square_at_one_one_l865_86533

theorem complex_square_at_one_one : 
  ∀ z : ℂ, (z.re = 1 ∧ z.im = 1) → z^2 = 2*I :=
by
  sorry

end complex_square_at_one_one_l865_86533


namespace conditional_probability_l865_86598

theorem conditional_probability (P_AB P_A : ℝ) (h1 : P_AB = 3/10) (h2 : P_A = 3/5) :
  P_AB / P_A = 1/2 := by
  sorry

end conditional_probability_l865_86598


namespace coffee_bread_combinations_l865_86580

theorem coffee_bread_combinations (coffee_types bread_types : ℕ) 
  (h1 : coffee_types = 2) (h2 : bread_types = 3) : 
  coffee_types * bread_types = 6 := by
  sorry

end coffee_bread_combinations_l865_86580


namespace min_reciprocal_sum_l865_86566

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) (hneq : x ≠ y) :
  1 / x + 1 / y > 1 / 3 := by
sorry

end min_reciprocal_sum_l865_86566


namespace isosceles_triangle_perimeter_l865_86575

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 8. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 2 ∧ b = 2 ∧ c = 4 →  -- Two sides are 2, one side is 4
  a + b > c →              -- Triangle inequality
  a = b →                  -- Isosceles condition
  a + b + c = 8 :=         -- Perimeter is 8
by
  sorry


end isosceles_triangle_perimeter_l865_86575


namespace fraction_problem_l865_86596

theorem fraction_problem (n d : ℕ) (h1 : d = 2*n - 1) (h2 : (n + 1) * 5 = (d + 1) * 3) : n = 5 ∧ d = 9 := by
  sorry

end fraction_problem_l865_86596


namespace max_percentage_offering_either_or_both_l865_86554

-- Define the percentage of companies offering wireless internet
def wireless_internet_percentage : ℚ := 20 / 100

-- Define the percentage of companies offering free snacks
def free_snacks_percentage : ℚ := 70 / 100

-- Theorem statement
theorem max_percentage_offering_either_or_both :
  ∃ (max_percentage : ℚ),
    max_percentage = wireless_internet_percentage + free_snacks_percentage ∧
    max_percentage ≤ 1 ∧
    ∀ (actual_percentage : ℚ),
      actual_percentage ≤ max_percentage :=
by sorry

end max_percentage_offering_either_or_both_l865_86554


namespace sequence_a_property_l865_86535

def sequence_a (n : ℕ) : ℚ :=
  1 / (n * (n + 1))

def S (n : ℕ) : ℚ :=
  n^2 * sequence_a n

theorem sequence_a_property :
  ∀ n : ℕ, n ≥ 1 →
    (sequence_a 1 = 1) ∧
    (S n = n^2 * sequence_a n) ∧
    (sequence_a n = 1 / (n * (n + 1))) :=
by sorry

end sequence_a_property_l865_86535


namespace expected_difference_l865_86591

/-- The number of students in the school -/
def total_students : ℕ := 100

/-- The number of classes and teachers -/
def num_classes : ℕ := 5

/-- The distribution of students across classes -/
def class_sizes : List ℕ := [40, 40, 10, 5, 5]

/-- The expected number of students per class when choosing a teacher at random -/
def t : ℚ := (total_students : ℚ) / num_classes

/-- The expected number of students per class when choosing a student at random -/
def s : ℚ := (List.sum (List.map (fun x => x * x) class_sizes) : ℚ) / total_students

theorem expected_difference :
  t - s = -27/2 := by sorry

end expected_difference_l865_86591


namespace sum_of_fractions_l865_86506

theorem sum_of_fractions (x y z : ℕ) : 
  (Nat.gcd x 9 = 1) → 
  (Nat.gcd y 15 = 1) → 
  (Nat.gcd z 14 = 1) → 
  (x * y * z : ℚ) / (9 * 15 * 14) = 1 / 6 → 
  x + y + z = 21 := by
sorry

end sum_of_fractions_l865_86506


namespace triangle_median_and_altitude_l865_86540

structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) := True

def isMedian (l : ℝ → ℝ → Prop) (A B C : Point) : Prop :=
  ∃ D : Point, D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2 ∧ l D.x D.y ∧ l A.x A.y

def isAltitude (l : ℝ → ℝ → Prop) (A B C : Point) : Prop :=
  ∀ x y : ℝ, l x y → (x - A.x) * (C.x - A.x) + (y - A.y) * (C.y - A.y) = 0

theorem triangle_median_and_altitude 
  (A B C : Point)
  (h_triangle : Triangle A B C)
  (h_A : A.x = 1 ∧ A.y = 3)
  (h_B : B.x = 5 ∧ B.y = 1)
  (h_C : C.x = -1 ∧ C.y = -1) :
  (isMedian (fun x y => 3 * x + y - 6 = 0) A B C) ∧
  (isAltitude (fun x y => x + 2 * y - 7 = 0) B A C) := by
  sorry

end triangle_median_and_altitude_l865_86540


namespace min_length_PQ_l865_86518

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a b c : ℝ) : Prop := b = (a + c) / 2

-- Define the moving line l
def line_l (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 2)

-- Define the line that Q lies on
def line_Q (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0

-- Define the minimum distance function
def min_distance (A P Q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem min_length_PQ (a b c : ℝ) :
  a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 →
  is_arithmetic_sequence a b c →
  ∃ (P Q : ℝ × ℝ),
    line_l a b c P.1 P.2 ∧
    line_Q Q.1 Q.2 ∧
    (∀ (P' Q' : ℝ × ℝ),
      line_l a b c P'.1 P'.2 →
      line_Q Q'.1 Q'.2 →
      min_distance point_A P Q ≤ min_distance point_A P' Q') →
    min_distance point_A P Q = 1 :=
sorry

end min_length_PQ_l865_86518


namespace monotone_increasing_inequalities_l865_86526

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Theorem statement
theorem monotone_increasing_inequalities 
  (h1 : ∀ x, f' x > 0) 
  (h2 : ∀ x, HasDerivAt f (f' x) x) 
  (x₁ x₂ : ℝ) 
  (h3 : x₁ ≠ x₂) : 
  (f x₁ - f x₂) * (x₁ - x₂) > 0 ∧ 
  (f x₁ - f x₂) * (x₂ - x₁) < 0 ∧ 
  (f x₂ - f x₁) * (x₂ - x₁) > 0 :=
by sorry

end monotone_increasing_inequalities_l865_86526


namespace triangle_properties_l865_86502

-- Define a triangle with given properties
structure Triangle where
  a : ℝ  -- side BC
  m : ℝ  -- altitude from B to AC
  k : ℝ  -- median to side AC
  a_pos : 0 < a
  m_pos : 0 < m
  k_pos : 0 < k

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  let b := 2 * Real.sqrt (t.k^2 + t.a * (t.a - Real.sqrt (4 * t.k^2 - t.m^2)))
  let c := 2 * Real.sqrt (t.k^2 + (t.a/2) * ((t.a/2) - Real.sqrt (4 * t.k^2 - t.m^2)))
  (∃ (γ β : ℝ),
    b > 0 ∧
    c > 0 ∧
    Real.sin γ = t.m / b ∧
    Real.sin β = t.m / c) := by
  sorry


end triangle_properties_l865_86502


namespace parallel_line_to_hyperbola_asymptote_l865_86568

/-- Given a hyperbola (x²/16) - (y²/9) = 1 and a line y = kx - 1 parallel to one of its asymptotes,
    where k > 0, prove that k = 3/4 -/
theorem parallel_line_to_hyperbola_asymptote
  (k : ℝ)
  (h1 : k > 0)
  (h2 : ∃ (x y : ℝ), y = k * x - 1 ∧ (x^2 / 16) - (y^2 / 9) = 1)
  (h3 : ∃ (m : ℝ), (∀ (x y : ℝ), y = m * x → (x^2 / 16) - (y^2 / 9) = 1) ∧
                   (∃ (b : ℝ), ∀ (x : ℝ), k * x - 1 = m * x + b)) :
  k = 3/4 := by
  sorry

end parallel_line_to_hyperbola_asymptote_l865_86568


namespace line_inclination_45_deg_l865_86529

/-- Given two points P(-2, m) and Q(m, 4) on a line with inclination angle 45°, prove m = 1 -/
theorem line_inclination_45_deg (m : ℝ) : 
  let P : ℝ × ℝ := (-2, m)
  let Q : ℝ × ℝ := (m, 4)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  slope = 1 → m = 1 := by
  sorry

end line_inclination_45_deg_l865_86529


namespace theta_range_l865_86539

theorem theta_range (θ : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12 :=
by sorry

end theta_range_l865_86539


namespace y_in_terms_of_x_l865_86550

theorem y_in_terms_of_x (m : ℕ) (x y : ℝ) 
  (hx : x = 2^m + 1) 
  (hy : y = 3 + 2^(m+1)) : 
  y = 2*x + 1 := by
  sorry

end y_in_terms_of_x_l865_86550


namespace subset_condition_main_result_l865_86534

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ a = 0 ∨ a = 1/3 ∨ a = 1/5 := by
  sorry

def solution_set : Set ℝ := {0, 1/3, 1/5}

theorem main_result : {a : ℝ | B a ⊆ A} = solution_set := by
  sorry

end subset_condition_main_result_l865_86534


namespace total_cans_collected_l865_86532

/-- The number of cans collected by LaDonna -/
def ladonna_cans : ℕ := 25

/-- The number of cans collected by Prikya -/
def prikya_cans : ℕ := 2 * ladonna_cans

/-- The number of cans collected by Yoki -/
def yoki_cans : ℕ := 10

/-- The total number of cans collected -/
def total_cans : ℕ := ladonna_cans + prikya_cans + yoki_cans

theorem total_cans_collected : total_cans = 85 := by
  sorry

end total_cans_collected_l865_86532


namespace trailer_homes_proof_l865_86553

/-- Represents the number of new trailer homes added -/
def new_homes : ℕ := 17

/-- Represents the initial number of trailer homes -/
def initial_homes : ℕ := 25

/-- Represents the initial average age of trailer homes in years -/
def initial_avg_age : ℚ := 15

/-- Represents the current average age of all trailer homes in years -/
def current_avg_age : ℚ := 12

/-- Represents the time elapsed since new homes were added, in years -/
def years_passed : ℕ := 3

theorem trailer_homes_proof :
  (initial_homes * (initial_avg_age + years_passed) + new_homes * years_passed) / 
  (initial_homes + new_homes) = current_avg_age :=
sorry

end trailer_homes_proof_l865_86553


namespace cubic_roots_l865_86508

def f (x : ℝ) : ℝ := x^3 - 4*x^2 - 7*x + 10

theorem cubic_roots :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 5) ∧
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 5 = 0) :=
sorry

end cubic_roots_l865_86508


namespace boys_usual_time_to_school_l865_86559

theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_time > 0 →
  usual_rate > 0 →
  (7/6 * usual_rate) * (usual_time - 2) = usual_rate * usual_time →
  usual_time = 14 := by
  sorry

end boys_usual_time_to_school_l865_86559


namespace square_sum_equals_90_l865_86578

theorem square_sum_equals_90 (x y : ℝ) 
  (h1 : x * (2 * x + y) = 18) 
  (h2 : y * (2 * x + y) = 72) : 
  (2 * x + y)^2 = 90 := by
sorry

end square_sum_equals_90_l865_86578


namespace cube_intersection_length_l865_86590

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with edge length a -/
structure Cube (a : ℝ) where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D
  F : Point3D
  G : Point3D
  H : Point3D

/-- The theorem to be proved -/
theorem cube_intersection_length (a : ℝ) (cube : Cube a) 
  (M : Point3D) (N : Point3D) (P : Point3D) (T : Point3D)
  (h_a : a > 0)
  (h_M : M.x = a ∧ M.y = a ∧ M.z = a/2)
  (h_N : N.x = a ∧ N.y = a/3 ∧ N.z = a)
  (h_P : P.x = 0 ∧ P.y = 0 ∧ P.z = 3*a/4)
  (h_T : T.x = 0 ∧ T.y = a ∧ 0 ≤ T.z ∧ T.z ≤ a)
  (h_plane : ∃ (k : ℝ), k * (M.x - P.x) * (N.y - P.y) * (T.z - P.z) = 
                         k * (N.x - P.x) * (M.y - P.y) * (T.z - P.z) + 
                         k * (T.x - P.x) * (M.y - P.y) * (N.z - P.z)) :
  ∃ (DT : ℝ), DT = 5*a/6 ∧ DT = Real.sqrt ((T.x - cube.D.x)^2 + (T.y - cube.D.y)^2 + (T.z - cube.D.z)^2) :=
sorry

end cube_intersection_length_l865_86590


namespace raghu_investment_l865_86501

/-- Proves that Raghu's investment is 2000 given the problem conditions --/
theorem raghu_investment (raghu trishul vishal : ℝ) : 
  trishul = raghu * 0.9 →
  vishal = trishul * 1.1 →
  raghu + trishul + vishal = 5780 →
  raghu = 2000 := by
sorry

end raghu_investment_l865_86501


namespace plumber_toilet_charge_l865_86585

def sink_charge : ℕ := 30
def shower_charge : ℕ := 40

def job1_earnings (toilet_charge : ℕ) : ℕ := 3 * toilet_charge + 3 * sink_charge
def job2_earnings (toilet_charge : ℕ) : ℕ := 2 * toilet_charge + 5 * sink_charge
def job3_earnings (toilet_charge : ℕ) : ℕ := toilet_charge + 2 * shower_charge + 3 * sink_charge

def max_earnings : ℕ := 250

theorem plumber_toilet_charge :
  ∃ (toilet_charge : ℕ),
    (job1_earnings toilet_charge ≤ max_earnings) ∧
    (job2_earnings toilet_charge ≤ max_earnings) ∧
    (job3_earnings toilet_charge ≤ max_earnings) ∧
    ((job1_earnings toilet_charge = max_earnings) ∨
     (job2_earnings toilet_charge = max_earnings) ∨
     (job3_earnings toilet_charge = max_earnings)) ∧
    toilet_charge = 50 :=
by sorry

end plumber_toilet_charge_l865_86585


namespace negative_eight_meters_westward_l865_86522

-- Define the direction type
inductive Direction
| East
| West

-- Define a function to convert meters to a direction and magnitude
def metersToDirection (x : ℤ) : Direction × ℕ :=
  if x ≥ 0 then
    (Direction.East, x.natAbs)
  else
    (Direction.West, (-x).natAbs)

-- State the theorem
theorem negative_eight_meters_westward :
  metersToDirection (-8) = (Direction.West, 8) :=
sorry

end negative_eight_meters_westward_l865_86522


namespace rectangle_perimeter_l865_86543

/-- Given a square with perimeter 180 units divided into 3 congruent rectangles,
    prove that the perimeter of one rectangle is 120 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (num_rectangles : ℕ) : 
  square_perimeter = 180 →
  num_rectangles = 3 →
  let square_side := square_perimeter / 4
  let rect_length := square_side
  let rect_width := square_side / num_rectangles
  2 * (rect_length + rect_width) = 120 :=
by sorry

end rectangle_perimeter_l865_86543


namespace scientific_notation_of_35_8_billion_l865_86521

theorem scientific_notation_of_35_8_billion : 
  (35800000000 : ℝ) = 3.58 * (10 : ℝ)^10 := by sorry

end scientific_notation_of_35_8_billion_l865_86521


namespace equation_unique_solution_l865_86503

theorem equation_unique_solution :
  ∃! x : ℝ, (Real.sqrt x + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 3*x) ∧ 
  (x = 400/49) := by
  sorry

end equation_unique_solution_l865_86503


namespace smallest_integer_in_sequence_l865_86597

theorem smallest_integer_in_sequence (a b c : ℤ) : 
  a < b ∧ b < c ∧ c < 90 →
  (a + b + c + 90) / 4 = 72 →
  a ≥ 21 :=
by sorry

end smallest_integer_in_sequence_l865_86597


namespace polygon_with_360_degree_sum_has_4_sides_l865_86520

theorem polygon_with_360_degree_sum_has_4_sides :
  ∀ (n : ℕ), n ≥ 3 →
  (n - 2) * 180 = 360 →
  n = 4 :=
by
  sorry

end polygon_with_360_degree_sum_has_4_sides_l865_86520


namespace building_floors_l865_86545

/-- The number of floors in a building given Earl's movements and position -/
theorem building_floors
  (P Q R S T X : ℕ)
  (h_x_lower : 1 < X)
  (h_x_upper : X < 50)
  : ∃ (F : ℕ), F = 1 + P - Q + R - S + T + X :=
by sorry

end building_floors_l865_86545


namespace chord_length_theorem_l865_86537

/-- The chord length theorem -/
theorem chord_length_theorem (m : ℝ) : 
  m > 0 → 
  (∃ (x y : ℝ), x - y + m = 0 ∧ (x - 1)^2 + (y - 1)^2 = 3) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ - y₁ + m = 0 ∧ (x₁ - 1)^2 + (y₁ - 1)^2 = 3 ∧
    x₂ - y₂ + m = 0 ∧ (x₂ - 1)^2 + (y₂ - 1)^2 = 3 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = m^2) →
  m = 2 :=
by sorry

end chord_length_theorem_l865_86537


namespace john_jury_duty_days_l865_86551

/-- The number of days John spends on jury duty -/
def jury_duty_days (jury_selection_days : ℕ) (trial_multiplier : ℕ) 
  (deliberation_days : ℕ) (deliberation_hours_per_day : ℕ) (hours_per_day : ℕ) : ℕ :=
  jury_selection_days + 
  (trial_multiplier * jury_selection_days) + 
  (deliberation_days * deliberation_hours_per_day) / hours_per_day

/-- Theorem stating that John spends 14 days on jury duty -/
theorem john_jury_duty_days : 
  jury_duty_days 2 4 6 16 24 = 14 := by
  sorry

end john_jury_duty_days_l865_86551


namespace P_intersect_Q_l865_86562

/-- The set P of vectors -/
def P : Set (ℝ × ℝ) := {a | ∃ m : ℝ, a = (1, 0) + m • (0, 1)}

/-- The set Q of vectors -/
def Q : Set (ℝ × ℝ) := {b | ∃ n : ℝ, b = (1, 1) + n • (-1, 1)}

/-- The theorem stating that the intersection of P and Q is the singleton set containing (1,1) -/
theorem P_intersect_Q : P ∩ Q = {(1, 1)} := by sorry

end P_intersect_Q_l865_86562


namespace existence_of_sequence_l865_86528

theorem existence_of_sequence (p q : ℝ) (y : Fin 2017 → ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hpq : p + q = 1) :
  ∃ x : Fin 2017 → ℝ, ∀ i : Fin 2017,
    p * max (x i) (x (i.succ)) + q * min (x i) (x (i.succ)) = y i :=
by sorry

end existence_of_sequence_l865_86528


namespace max_value_of_function_l865_86542

theorem max_value_of_function : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x) - Real.cos (2 * x + π / 6)
  ∃ (M : ℝ), M = Real.sqrt 3 ∧ ∀ x, f x ≤ M ∧ ∃ x₀, f x₀ = M :=
by sorry

end max_value_of_function_l865_86542


namespace special_factors_count_l865_86546

/-- A function that returns the number of positive factors of 60 that are multiples of 5 but not multiples of 3 -/
def count_special_factors : ℕ :=
  (Finset.filter (fun n => 60 % n = 0 ∧ n % 5 = 0 ∧ n % 3 ≠ 0) (Finset.range 61)).card

/-- Theorem stating that the number of positive factors of 60 that are multiples of 5 but not multiples of 3 is 2 -/
theorem special_factors_count : count_special_factors = 2 := by
  sorry

end special_factors_count_l865_86546


namespace discount_equivalence_l865_86593

/-- Proves that a 30% discount followed by a 15% discount is equivalent to a 40.5% single discount -/
theorem discount_equivalence (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.30
  let second_discount := 0.15
  let discounted_price := original_price * (1 - first_discount)
  let final_price := discounted_price * (1 - second_discount)
  let equivalent_discount := 1 - (final_price / original_price)
  equivalent_discount = 0.405 := by
sorry

end discount_equivalence_l865_86593


namespace work_completion_time_l865_86584

/-- The number of days y needs to finish the work alone -/
def y_days : ℕ := 24

/-- The number of days y worked before leaving -/
def y_worked : ℕ := 12

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℕ := 18

/-- The number of days x needs to finish the work alone -/
def x_days : ℕ := 36

theorem work_completion_time :
  x_days = 36 :=
by sorry

end work_completion_time_l865_86584


namespace root_product_l865_86564

theorem root_product (a b : ℝ) : 
  (a^2 + 2*a - 2023 = 0) → 
  (b^2 + 2*b - 2023 = 0) → 
  (a + 1) * (b + 1) = -2024 := by
sorry

end root_product_l865_86564


namespace number_equality_l865_86576

theorem number_equality (x : ℝ) : (2 * x + 20 = 8 * x - 4) ↔ (x = 4) := by
  sorry

end number_equality_l865_86576


namespace shower_tiles_count_l865_86507

/-- Calculates the total number of tiles in a 3-sided shower --/
def shower_tiles (sides : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  sides * width * height

/-- Theorem: The total number of tiles in a 3-sided shower with 8 tiles in width and 20 tiles in height is 480 --/
theorem shower_tiles_count : shower_tiles 3 8 20 = 480 := by
  sorry

end shower_tiles_count_l865_86507


namespace volume_ratio_of_cubes_l865_86579

/-- The perimeter of a square face of a cube -/
def face_perimeter (s : ℝ) : ℝ := 4 * s

/-- The volume of a cube -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- Theorem: Given two cubes A and B with face perimeters 40 cm and 64 cm respectively, 
    the ratio of their volumes is 125:512 -/
theorem volume_ratio_of_cubes (s_A s_B : ℝ) 
  (h_A : face_perimeter s_A = 40)
  (h_B : face_perimeter s_B = 64) : 
  (cube_volume s_A) / (cube_volume s_B) = 125 / 512 := by
  sorry

end volume_ratio_of_cubes_l865_86579


namespace dorothy_sea_glass_count_l865_86516

-- Define the sea glass counts for Blanche and Rose
def blanche_green : ℕ := 12
def blanche_red : ℕ := 3
def rose_red : ℕ := 9
def rose_blue : ℕ := 11

-- Define Dorothy's sea glass counts based on the conditions
def dorothy_red : ℕ := 2 * (blanche_red + rose_red)
def dorothy_blue : ℕ := 3 * rose_blue

-- Define Dorothy's total sea glass count
def dorothy_total : ℕ := dorothy_red + dorothy_blue

-- Theorem to prove
theorem dorothy_sea_glass_count : dorothy_total = 57 := by
  sorry

end dorothy_sea_glass_count_l865_86516


namespace max_sum_of_factors_l865_86527

theorem max_sum_of_factors (A B C : ℕ) : 
  A ≠ B → B ≠ C → A ≠ C → 
  A > 0 → B > 0 → C > 0 → 
  A * B * C = 2310 → 
  A + B + C ≤ 52 := by
sorry

end max_sum_of_factors_l865_86527


namespace pedestrian_speed_theorem_l865_86569

/-- Given two pedestrians moving in the same direction, this theorem proves
    that the speed of the second pedestrian is either 6 m/s or 20/3 m/s,
    given the initial conditions. -/
theorem pedestrian_speed_theorem 
  (S₀ : ℝ) (v₁ : ℝ) (t : ℝ) (S : ℝ)
  (h₁ : S₀ = 200) 
  (h₂ : v₁ = 7)
  (h₃ : t = 5 * 60) -- 5 minutes in seconds
  (h₄ : S = 100) :
  ∃ v₂ : ℝ, (v₂ = 6 ∨ v₂ = 20/3) ∧ 
  (S₀ - S = (v₁ - v₂) * t) :=
by sorry

end pedestrian_speed_theorem_l865_86569


namespace parabola_shift_theorem_l865_86561

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + dy }

theorem parabola_shift_theorem :
  let initial_parabola : Parabola := { a := -1, h := 1, k := 2 }
  let shifted_parabola := shift_horizontal (shift_vertical initial_parabola 2) 1
  shifted_parabola = { a := -1, h := 0, k := 4 } := by sorry

end parabola_shift_theorem_l865_86561


namespace cube_inequality_l865_86587

theorem cube_inequality (a b : ℝ) (h : a < 0 ∧ 0 < b) : a^3 < b^3 := by
  sorry

end cube_inequality_l865_86587


namespace bryan_work_hours_l865_86530

/-- Represents Bryan's daily work schedule --/
structure WorkSchedule where
  customer_outreach : ℝ
  advertisement : ℝ
  marketing : ℝ

/-- Calculates the total working hours given a work schedule --/
def total_hours (schedule : WorkSchedule) : ℝ :=
  schedule.customer_outreach + schedule.advertisement + schedule.marketing

/-- Theorem stating Bryan's total working hours --/
theorem bryan_work_hours :
  ∀ (schedule : WorkSchedule),
    schedule.customer_outreach = 4 →
    schedule.advertisement = schedule.customer_outreach / 2 →
    schedule.marketing = 2 →
    total_hours schedule = 8 := by
  sorry

end bryan_work_hours_l865_86530


namespace phi_value_l865_86570

/-- Given a function f and constants ω and φ, proves that φ = π/6 under certain conditions. -/
theorem phi_value (f g : ℝ → ℝ) (ω φ : ℝ) : 
  (∀ x, f x = 2 * Real.sin (ω * x + φ)) →
  ω > 0 →
  |φ| < π / 2 →
  (∀ x, f (x + π) = f x) →
  (∀ x, f (x - π / 6) = g x) →
  (∀ x, g (x + π / 3) = g (π / 3 - x)) →
  φ = π / 6 := by
sorry


end phi_value_l865_86570


namespace ted_fruit_purchase_l865_86582

/-- The number of bananas Ted needs to purchase -/
def num_bananas : ℕ := 5

/-- The cost of one banana in dollars -/
def banana_cost : ℚ := 2

/-- The cost of one orange in dollars -/
def orange_cost : ℚ := 3/2

/-- The total amount Ted needs to spend on fruits in dollars -/
def total_cost : ℚ := 25

/-- The number of oranges Ted needs to purchase -/
def num_oranges : ℕ := 10

theorem ted_fruit_purchase :
  (num_bananas : ℚ) * banana_cost + (num_oranges : ℚ) * orange_cost = total_cost :=
sorry

end ted_fruit_purchase_l865_86582


namespace road_trip_time_calculation_l865_86556

theorem road_trip_time_calculation (dist_wa_id : ℝ) (dist_id_nv : ℝ) (speed_wa_id : ℝ) (speed_id_nv : ℝ)
  (h1 : dist_wa_id = 640)
  (h2 : dist_id_nv = 550)
  (h3 : speed_wa_id = 80)
  (h4 : speed_id_nv = 50)
  (h5 : speed_wa_id > 0)
  (h6 : speed_id_nv > 0) :
  dist_wa_id / speed_wa_id + dist_id_nv / speed_id_nv = 19 := by
  sorry

end road_trip_time_calculation_l865_86556


namespace boat_distance_along_stream_l865_86567

/-- Represents the speed of a boat in km/hr -/
def boat_speed : ℝ := 6

/-- Represents the distance traveled against the stream in km -/
def distance_against : ℝ := 5

/-- Represents the time of travel in hours -/
def travel_time : ℝ := 1

/-- Calculates the speed of the stream based on the boat's speed and distance traveled against the stream -/
def stream_speed : ℝ := boat_speed - distance_against

/-- Calculates the effective speed of the boat along the stream -/
def effective_speed : ℝ := boat_speed + stream_speed

/-- Theorem: The boat travels 7 km along the stream in one hour -/
theorem boat_distance_along_stream :
  effective_speed * travel_time = 7 := by sorry

end boat_distance_along_stream_l865_86567


namespace infinite_triples_theorem_l865_86523

def is_sum_of_two_squares (n : ℕ) : Prop := ∃ a b : ℕ, n = a^2 + b^2

theorem infinite_triples_theorem :
  (∃ f : ℕ → ℕ, ∀ m : ℕ,
    is_sum_of_two_squares (2 * 100^(f m)) ∧
    ¬is_sum_of_two_squares (2 * 100^(f m) - 1) ∧
    ¬is_sum_of_two_squares (2 * 100^(f m) + 1)) ∧
  (∃ g : ℕ → ℕ, ∀ m : ℕ,
    is_sum_of_two_squares (2 * (g m^2 - g m)^2 + 1) ∧
    is_sum_of_two_squares (2 * (g m^2 - g m)^2) ∧
    is_sum_of_two_squares (2 * (g m^2 - g m)^2 + 2)) :=
by sorry

end infinite_triples_theorem_l865_86523


namespace abs_inequality_range_l865_86560

theorem abs_inequality_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = |x - 2| + |x + a|) →
  (∀ x : ℝ, f x ≥ 3) →
  (a ≤ -5 ∨ a ≥ 1) :=
sorry

end abs_inequality_range_l865_86560


namespace function_increment_proof_l865_86504

/-- The function f(x) = 2x^2 + 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The initial x value -/
def x₀ : ℝ := 1

/-- The final x value -/
def x₁ : ℝ := 1.02

/-- The increment of x -/
def Δx : ℝ := x₁ - x₀

theorem function_increment_proof :
  f x₁ - f x₀ = 0.0808 :=
sorry

end function_increment_proof_l865_86504


namespace seasonal_work_term_l865_86500

/-- The established term of work for two seasonal workers -/
theorem seasonal_work_term (a r s : ℝ) (hr : r > 0) (hs : s > r) :
  ∃ x : ℝ, x > 0 ∧
  (x - a) * (s / (x + a)) = (x + a) * (r / (x - a)) ∧
  x = a * (s + r) / (s - r) := by
  sorry

end seasonal_work_term_l865_86500


namespace intersection_implies_a_value_l865_86581

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A a) ∩ (B a) = {-3} → a = -1 := by
  sorry

end intersection_implies_a_value_l865_86581


namespace symmetric_points_product_l865_86524

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are opposites of each other -/
def symmetric_wrt_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_product (a b : ℝ) :
  symmetric_wrt_x_axis (2, a) (b + 1, 3) → a * b = -3 := by
  sorry

#check symmetric_points_product

end symmetric_points_product_l865_86524


namespace painted_cube_probability_l865_86519

/-- Represents a 5x5x5 cube with three faces sharing a vertex painted -/
structure PaintedCube where
  size : ℕ
  painted_faces : ℕ
  size_eq : size = 5
  faces_eq : painted_faces = 3

/-- The number of unit cubes with exactly three painted faces -/
def num_three_painted (cube : PaintedCube) : ℕ := 8

/-- The number of unit cubes with exactly one painted face -/
def num_one_painted (cube : PaintedCube) : ℕ := 3 * (cube.size - 2)^2

/-- The total number of unit cubes in the large cube -/
def total_cubes (cube : PaintedCube) : ℕ := cube.size^3

/-- The number of ways to choose two cubes from the total -/
def total_combinations (cube : PaintedCube) : ℕ := (total_cubes cube).choose 2

/-- The number of ways to choose one cube with three painted faces and one with one painted face -/
def favorable_outcomes (cube : PaintedCube) : ℕ := (num_three_painted cube) * (num_one_painted cube)

/-- The probability of selecting one cube with three painted faces and one with one painted face -/
def probability (cube : PaintedCube) : ℚ :=
  (favorable_outcomes cube : ℚ) / (total_combinations cube : ℚ)

theorem painted_cube_probability (cube : PaintedCube) :
  probability cube = 24 / 875 := by sorry

end painted_cube_probability_l865_86519


namespace circle_rectangle_area_difference_l865_86599

/-- Given a rectangle with diagonal 10 and length-to-width ratio 2:1, and a circle with radius 5,
    prove that the difference between the circle's area and the rectangle's area is 25π - 40. -/
theorem circle_rectangle_area_difference :
  let rectangle_diagonal : ℝ := 10
  let length_width_ratio : ℚ := 2 / 1
  let circle_radius : ℝ := 5
  let rectangle_width : ℝ := (rectangle_diagonal ^ 2 / (1 + length_width_ratio ^ 2)) ^ (1 / 2 : ℝ)
  let rectangle_length : ℝ := length_width_ratio * rectangle_width
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  let circle_area : ℝ := π * circle_radius ^ 2
  circle_area - rectangle_area = 25 * π - 40 := by
  sorry

end circle_rectangle_area_difference_l865_86599


namespace no_real_roots_quadratic_l865_86538

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x - k ≠ 0) → k < -9/4 := by
  sorry

end no_real_roots_quadratic_l865_86538


namespace num_sandwich_combinations_l865_86515

/-- Represents the number of different types of bread available. -/
def num_bread : ℕ := 3

/-- Represents the number of different types of meat available. -/
def num_meat : ℕ := 5

/-- Represents the number of different types of cheese available. -/
def num_cheese : ℕ := 4

/-- Represents the number of sandwiches excluded due to the turkey/swiss cheese combination. -/
def turkey_swiss_exclusions : ℕ := num_bread

/-- Represents the number of sandwiches excluded due to the roast beef/rye bread combination. -/
def roast_beef_rye_exclusions : ℕ := num_cheese

/-- Calculates the total number of possible sandwich combinations without restrictions. -/
def total_combinations : ℕ := num_bread * num_meat * num_cheese

/-- Theorem stating that the number of different sandwiches that can be ordered is 53. -/
theorem num_sandwich_combinations : 
  total_combinations - turkey_swiss_exclusions - roast_beef_rye_exclusions = 53 := by
  sorry

end num_sandwich_combinations_l865_86515


namespace last_digit_89_base_5_l865_86592

theorem last_digit_89_base_5 : 89 % 5 = 4 := by
  sorry

end last_digit_89_base_5_l865_86592


namespace section_B_avg_weight_l865_86517

def section_A_students : ℕ := 50
def section_B_students : ℕ := 40
def total_students : ℕ := section_A_students + section_B_students
def section_A_avg_weight : ℝ := 50
def total_avg_weight : ℝ := 58.89

theorem section_B_avg_weight :
  let section_B_weight := total_students * total_avg_weight - section_A_students * section_A_avg_weight
  section_B_weight / section_B_students = 70.0025 := by
sorry

end section_B_avg_weight_l865_86517


namespace product_ratio_theorem_l865_86586

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1/2 := by
  sorry

end product_ratio_theorem_l865_86586


namespace multiple_properties_l865_86512

theorem multiple_properties (a b c : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k)
  (hb : ∃ k : ℤ, b = 12 * k)
  (hc : ∃ k : ℤ, c = 9 * k) :
  (∃ k : ℤ, b = 3 * k) ∧
  (∃ k : ℤ, a - b = 3 * k) ∧
  (∃ k : ℤ, a - c = 3 * k) ∧
  (∃ k : ℤ, c - b = 3 * k) := by
  sorry

end multiple_properties_l865_86512


namespace intersection_of_A_and_B_l865_86558

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 6*x + 8 < 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l865_86558


namespace exam_score_exam_score_specific_case_l865_86574

theorem exam_score (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : ℕ :=
  let wrong_answers := total_questions - correct_answers
  let total_marks := correct_answers * marks_per_correct - wrong_answers * marks_lost_per_wrong
  total_marks

theorem exam_score_specific_case : exam_score 75 40 4 1 = 125 := by
  sorry

end exam_score_exam_score_specific_case_l865_86574


namespace total_pizzas_is_fifteen_l865_86541

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := 6

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := lunch_pizzas + dinner_pizzas

/-- Theorem: The total number of pizzas served today is 15 -/
theorem total_pizzas_is_fifteen : total_pizzas = 15 := by sorry

end total_pizzas_is_fifteen_l865_86541


namespace theater_seats_l865_86563

theorem theater_seats (adult_price child_price total_income num_children : ℚ)
  (h1 : adult_price = 3)
  (h2 : child_price = 3/2)
  (h3 : total_income = 510)
  (h4 : num_children = 60)
  (h5 : ∃ num_adults : ℚ, num_adults * adult_price + num_children * child_price = total_income) :
  ∃ total_seats : ℚ, total_seats = num_children + (total_income - num_children * child_price) / adult_price ∧ total_seats = 200 := by
  sorry

end theater_seats_l865_86563


namespace snack_slices_theorem_l865_86511

/-- Represents the household bread consumption scenario -/
structure HouseholdBread where
  members : ℕ
  breakfast_slices_per_member : ℕ
  slices_per_loaf : ℕ
  loaves_consumed : ℕ
  days_lasted : ℕ

/-- Calculate the number of slices each member consumes for snacks daily -/
def snack_slices_per_member_per_day (hb : HouseholdBread) : ℕ :=
  let total_slices := hb.loaves_consumed * hb.slices_per_loaf
  let breakfast_slices := hb.members * hb.breakfast_slices_per_member * hb.days_lasted
  let snack_slices := total_slices - breakfast_slices
  snack_slices / (hb.members * hb.days_lasted)

/-- Theorem stating that each member consumes 2 slices of bread for snacks daily -/
theorem snack_slices_theorem (hb : HouseholdBread) 
  (h1 : hb.members = 4)
  (h2 : hb.breakfast_slices_per_member = 3)
  (h3 : hb.slices_per_loaf = 12)
  (h4 : hb.loaves_consumed = 5)
  (h5 : hb.days_lasted = 3) :
  snack_slices_per_member_per_day hb = 2 := by
  sorry


end snack_slices_theorem_l865_86511


namespace geometric_sequence_sum_l865_86588

theorem geometric_sequence_sum (a r : ℝ) : 
  a + a * r = 7 →
  a * (r^6 - 1) / (r - 1) = 91 →
  a + a * r + a * r^2 + a * r^3 = 28 := by
sorry

end geometric_sequence_sum_l865_86588


namespace rook_placements_count_l865_86595

/-- Represents a special chessboard with a long horizontal row at the bottom -/
structure SpecialChessboard where
  rows : Nat
  columns : Nat
  longRowLength : Nat

/-- Represents a rook placement on the special chessboard -/
structure RookPlacement where
  row : Nat
  column : Nat

/-- Checks if two rook placements attack each other on the special chessboard -/
def attacks (board : SpecialChessboard) (r1 r2 : RookPlacement) : Prop :=
  r1.row = r2.row ∨ r1.column = r2.column

/-- Counts the number of valid ways to place 3 rooks on the special chessboard -/
def countValidPlacements (board : SpecialChessboard) : Nat :=
  sorry

/-- The main theorem stating that there are 168 ways to place 3 rooks on the special chessboard -/
theorem rook_placements_count (board : SpecialChessboard) 
  (h1 : board.rows = 4) 
  (h2 : board.columns = 8) 
  (h3 : board.longRowLength = 8) : 
  countValidPlacements board = 168 := by
  sorry

end rook_placements_count_l865_86595


namespace diagonal_length_is_sqrt_457_l865_86509

/-- An isosceles trapezoid with specific side lengths -/
structure IsoscelesTrapezoid :=
  (A B C D : ℝ × ℝ)
  (ab_length : dist A B = 24)
  (bc_length : dist B C = 13)
  (cd_length : dist C D = 12)
  (da_length : dist D A = 13)
  (isosceles : dist B C = dist D A)

/-- The length of the diagonal AC in the isosceles trapezoid -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  dist t.A t.C

/-- Theorem stating that the diagonal length is √457 -/
theorem diagonal_length_is_sqrt_457 (t : IsoscelesTrapezoid) :
  diagonal_length t = Real.sqrt 457 := by
  sorry


end diagonal_length_is_sqrt_457_l865_86509


namespace min_distance_between_curves_l865_86577

/-- The minimum distance between two points on different curves with the same y-coordinate --/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  min_dist = 3/2 ∧
  ∀ (a : ℝ) (x₁ x₂ : ℝ),
    a = 2 * (x₁ + 1) →
    a = x₂ + Real.log x₂ →
    |x₂ - x₁| ≥ min_dist :=
by sorry

end min_distance_between_curves_l865_86577


namespace complex_equation_sum_l865_86583

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (a + i) * i = b + i) : a + b = 0 := by
  sorry

end complex_equation_sum_l865_86583


namespace division_problem_l865_86544

theorem division_problem (x : ℝ) : (x / 0.08 = 800) → x = 64 := by
  sorry

end division_problem_l865_86544


namespace coprime_polynomials_l865_86552

theorem coprime_polynomials (n : ℕ) : 
  Nat.gcd (n^5 + 4*n^3 + 3*n) (n^4 + 3*n^2 + 1) = 1 := by
  sorry

end coprime_polynomials_l865_86552


namespace julia_stairs_difference_l865_86514

theorem julia_stairs_difference (jonny_stairs julia_stairs : ℕ) : 
  jonny_stairs = 1269 →
  julia_stairs < jonny_stairs / 3 →
  jonny_stairs + julia_stairs = 1685 →
  (jonny_stairs / 3) - julia_stairs = 7 :=
by
  sorry

end julia_stairs_difference_l865_86514


namespace jacket_pricing_l865_86557

theorem jacket_pricing (x : ℝ) : 
  (0.8 * (1 + 0.5) * x = x + 28) ↔ 
  (∃ (markup : ℝ) (discount : ℝ) (profit : ℝ), 
    markup = 0.5 ∧ 
    discount = 0.2 ∧ 
    profit = 28 ∧ 
    (1 - discount) * (1 + markup) * x - x = profit) :=
by sorry

end jacket_pricing_l865_86557


namespace even_function_interval_sum_zero_l865_86536

/-- A function f is even on an interval [a, b] if for all x in [a, b], f(x) = f(-x) -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, f x = f (-x)

/-- If f is an even function on the interval [a, b], then a + b = 0 -/
theorem even_function_interval_sum_zero (f : ℝ → ℝ) (a b : ℝ) 
  (h : IsEvenOn f a b) : a + b = 0 := by
  sorry

end even_function_interval_sum_zero_l865_86536


namespace eggs_in_park_l865_86548

theorem eggs_in_park (total_eggs club_house_eggs town_hall_eggs : ℕ) 
  (h1 : total_eggs = 20)
  (h2 : club_house_eggs = 12)
  (h3 : town_hall_eggs = 3) :
  total_eggs - club_house_eggs - town_hall_eggs = 5 :=
by
  sorry

end eggs_in_park_l865_86548


namespace fraction_pair_sum_equality_l865_86510

theorem fraction_pair_sum_equality (n : ℕ) (h : n > 2009) :
  ∃ (a b c d : ℕ), a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (1 : ℚ) / (n + 1 - a) + (1 : ℚ) / (n + 1 - b) =
  (1 : ℚ) / (n + 1 - c) + (1 : ℚ) / (n + 1 - d) :=
by sorry

end fraction_pair_sum_equality_l865_86510


namespace union_of_M_and_N_l865_86547

def M : Set ℝ := {x | x^2 - 6*x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by sorry

end union_of_M_and_N_l865_86547


namespace expression_approximation_l865_86589

theorem expression_approximation :
  let x := ((69.28 * 0.004)^3 * Real.sin (Real.pi/3)) / (0.03^2 * Real.log 0.58 * Real.cos (Real.pi/4))
  ∃ ε > 0, |x + 37.644| < ε ∧ ε < 0.001 := by
  sorry

end expression_approximation_l865_86589


namespace inequality_proof_l865_86565

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * b * c) + 1 ≥ 3 * (1 / (a^2 + b^2 + c^2) + 1 / (a + b + c)) := by
  sorry

end inequality_proof_l865_86565
