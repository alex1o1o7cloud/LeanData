import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l826_82629

open Real

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + m + 3
noncomputable def g (x : ℝ) : ℝ := 2^(x - 2)

-- State the theorem
theorem m_range_theorem (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) →
  -4 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l826_82629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l826_82683

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

-- Define set B
def B : Set ℝ := {x | x < 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l826_82683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_arccot_2A_squared_equals_pi_over_6_l826_82667

def A : ℕ → ℝ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 3
  | n+3 => 4 * A (n+2) - A (n+1)

theorem sum_arccot_2A_squared_equals_pi_over_6 :
  ∑' n, Real.arctan (1 / (2 * A n ^ 2)) = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_arccot_2A_squared_equals_pi_over_6_l826_82667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_megatek_manufacturing_percentage_l826_82636

/-- Represents the number of degrees in a full circle. -/
noncomputable def full_circle : ℝ := 360

/-- Represents the number of degrees in the manufacturing department's sector. -/
noncomputable def manufacturing_sector : ℝ := 18

/-- Calculates the percentage of employees in the manufacturing department. -/
noncomputable def manufacturing_percentage : ℝ := (manufacturing_sector / full_circle) * 100

/-- Theorem stating that the percentage of Megatek employees in the manufacturing department is 5%. -/
theorem megatek_manufacturing_percentage :
  manufacturing_percentage = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_megatek_manufacturing_percentage_l826_82636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_games_correct_l826_82674

/-- Represents the result of a single game -/
inductive GameResult
| A_Wins
| B_Wins

/-- Represents the state of the match -/
structure MatchState :=
  (a_score : ℕ)
  (b_score : ℕ)
  (games_played : ℕ)

/-- Checks if the match is over -/
def is_match_over (state : MatchState) : Bool :=
  (state.a_score ≥ state.b_score + 2) ∨ 
  (state.b_score ≥ state.a_score + 2) ∨ 
  (state.games_played = 6)

/-- Probability of A winning a single game -/
noncomputable def p_a_wins : ℝ := 2/3

/-- Probability of B winning a single game -/
noncomputable def p_b_wins : ℝ := 1/3

/-- Expected number of games played -/
noncomputable def expected_games : ℝ := 266/81

theorem expected_games_correct :
  expected_games = 266/81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_games_correct_l826_82674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_center_sum_l826_82650

/-- The origin -/
def O : ℝ × ℝ × ℝ := (0, 0, 0)

/-- A fixed point not on the origin -/
def fixed_point (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- Assumption that (a, b, c) is not the origin -/
axiom not_origin (a b c : ℝ) : fixed_point a b c ≠ O

/-- The plane intersects x-axis at (2a, 0, 0) -/
def A (a : ℝ) : ℝ × ℝ × ℝ := (2*a, 0, 0)

/-- The plane intersects y-axis at (0, 2b, 0) -/
def B (b : ℝ) : ℝ × ℝ × ℝ := (0, 2*b, 0)

/-- The plane intersects z-axis at (0, 0, 2c) -/
def C (c : ℝ) : ℝ × ℝ × ℝ := (0, 0, 2*c)

/-- The center of the sphere passing through O, A, B, and C -/
def sphere_center (p q r : ℝ) : ℝ × ℝ × ℝ := (p, q, r)

/-- The theorem to be proved -/
theorem sphere_center_sum (a b c p q r : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) : 
  a / p + b / q + c / r = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_center_sum_l826_82650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_value_l826_82618

theorem min_cos_value (x : ℝ) (h : x ∈ Set.Icc (π/6) ((2*π)/3)) :
  ∃ (y : ℝ), y = Real.cos (x - π/8) ∧ 
  (∀ z ∈ Set.Icc (π/6) ((2*π)/3), Real.cos (z - π/8) ≥ y) ∧
  y = Real.cos (13*π/24) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_value_l826_82618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l826_82624

/-- The distance between two parallel lines ax + by + m = 0 and ax + by + n = 0 -/
noncomputable def distance_parallel_lines (a b m n : ℝ) : ℝ :=
  abs (m - n) / Real.sqrt (a^2 + b^2)

/-- The first line equation: 5x + 12y + 3 = 0 -/
def line1 (x y : ℝ) : Prop :=
  5 * x + 12 * y + 3 = 0

/-- The second line equation: 5x + 12y + 5 = 0 -/
def line2 (x y : ℝ) : Prop :=
  5 * x + 12 * y + 5 = 0

theorem distance_between_given_lines :
  distance_parallel_lines 5 12 3 5 = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l826_82624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_factorial_even_divisors_l826_82685

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- A function to count the number of even divisors of a natural number -/
def count_even_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d => Even d ∧ n % d = 0) (Finset.range (n + 1))).card

/-- Theorem stating that 8! has 84 even divisors -/
theorem eight_factorial_even_divisors :
  count_even_divisors (factorial 8) = 84 := by
  sorry

#eval count_even_divisors (factorial 8)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_factorial_even_divisors_l826_82685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_properties_l826_82613

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 3 * (Real.sin x)^2 - (Real.cos x)^2 + 2

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C

-- State the theorem
theorem triangle_function_properties 
  (t : Triangle) 
  (h1 : t.b / t.a = Real.sqrt 3) 
  (h2 : Real.sin (2 * t.A + t.C) / Real.sin t.A = 2 + 2 * Real.cos (t.A + t.C)) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-1) 2) ∧ 
  f t.B = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_function_properties_l826_82613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bagel_store_spending_l826_82600

theorem bagel_store_spending (b d : ℝ) :
  d = 0.7 * b →
  b = d + 15 →
  (b + d) * 1.1 = 93.5 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bagel_store_spending_l826_82600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_H_points_l826_82602

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def l (x : ℝ) : Prop := x = 4

-- Define the H point property
def is_H_point (x y : ℝ) : Prop :=
  C x y ∧ 
  ∃ (x_A y_A x_B : ℝ),
    C x_A y_A ∧ 
    l x_B ∧
    ((x - x_A)^2 + (y - y_A)^2 = (x - x_B)^2 + (y - 4)^2 ∨
     (x - x_A)^2 + (y - y_A)^2 = (x_A - x_B)^2 + (y_A - 4)^2)

-- Theorem statement
theorem infinitely_many_H_points :
  ∃ (S : Set ℝ), 
    (∀ x ∈ S, x ∈ Set.Icc (-2) 0 ∪ Set.Icc 1 2) ∧
    (Set.Infinite S) ∧
    (∀ x ∈ S, ∃ y, is_H_point x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_H_points_l826_82602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_store_ratio_l826_82621

/-- Given the conditions of the pet store inventory, prove the ratio of cats to dogs --/
theorem pet_store_ratio :
  ∀ (num_dogs num_cats num_birds num_fish : ℕ),
  num_dogs = 6 →
  num_birds = 2 * num_dogs →
  num_fish = 3 * num_dogs →
  num_cats + num_dogs + num_birds + num_fish = 39 →
  2 * num_cats = num_dogs := by
  sorry

#check pet_store_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_store_ratio_l826_82621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_status_verbal_conclusion_l826_82653

-- Define the nominal portion weight
def nominal_weight : ℝ := 390

-- Define the greatest deviation
def greatest_deviation : ℝ := 39

-- Define the condition that the greatest deviation doesn't exceed 10% of nominal weight
axiom deviation_within_limit : greatest_deviation ≤ 0.1 * nominal_weight

-- Define the condition for unreadable measurements
axiom unreadable_measurements_deviation (x : ℝ) : 
  x < greatest_deviation → x ≤ greatest_deviation

-- Define the standard deviation
def standard_deviation : ℝ := sorry

-- Define the condition that standard deviation doesn't exceed greatest deviation
axiom standard_deviation_bound : standard_deviation ≤ greatest_deviation

-- Define a proposition for the machine status
def machine_doesnt_require_repair : Prop := standard_deviation ≤ greatest_deviation

-- Theorem to prove
theorem machine_status : machine_doesnt_require_repair := by
  sorry

-- Additional theorem to connect the mathematical result to the verbal conclusion
theorem verbal_conclusion : 
  machine_doesnt_require_repair → "The machine does not require repair" = "The machine does not require repair" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_status_verbal_conclusion_l826_82653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l826_82661

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (Real.cos (C / 2) = Real.sqrt 5 / 3) →
  (a * Real.cos B + b * Real.cos A = 2) →
  -- Definition of triangle area using sine formula
  (∀ S : ℝ, S = 1/2 * a * b * Real.sin C) →
  -- Theorem to prove
  (∃ S : ℝ, S ≤ Real.sqrt 5 / 2 ∧ 
    ∀ S' : ℝ, (S' = 1/2 * a * b * Real.sin C) → S' ≤ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l826_82661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l826_82654

/-- Represents a triangle ABC with a point X on side AB -/
structure TriangleWithPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  X : ℝ × ℝ
  on_side : X.1 ∈ Set.Icc A.1 B.1 ∧ X.2 = A.2

/-- The angle bisector theorem -/
axiom angle_bisector_theorem {t : TriangleWithPoint} :
  (t.C.1 - t.X.1) * (t.B.2 - t.X.2) = (t.C.2 - t.X.2) * (t.B.1 - t.X.1) →
  ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) / ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) =
  (t.A.1 - t.X.1) / (t.B.1 - t.X.1)

theorem triangle_segment_length (t : TriangleWithPoint) 
  (bisects : (t.C.1 - t.X.1) * (t.B.2 - t.X.2) = (t.C.2 - t.X.2) * (t.B.1 - t.X.1))
  (ac_length : (t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2 = 24^2)
  (bc_length : (t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 = 36^2)
  (bx_length : (t.B.1 - t.X.1)^2 = 42^2) :
  (t.A.1 - t.X.1)^2 = 28^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l826_82654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_solutions_l826_82670

-- Define the piecewise function h(x)
noncomputable def h (x : ℝ) : ℝ :=
  if x < 0 then 5 * x + 10 else 3 * x - 18

-- Theorem statement
theorem h_solutions (x : ℝ) : h x = 5 ↔ x = -1 ∨ x = 23 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_solutions_l826_82670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l826_82626

theorem find_n : ∃ n : ℚ, (5 : ℝ) ^ (5 * n : ℝ) = (1 / 5 : ℝ) ^ ((2 * n - 15) : ℝ) ∧ n = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l826_82626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l826_82696

-- Define the sets A and B
def A : Set ℝ := {x | abs x < 3}
def B : Set ℝ := {x | Real.rpow 2 x > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l826_82696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_place_distance_with_obstacles_l826_82601

/-- Calculates the distance to a place given rowing conditions -/
noncomputable def calculate_distance (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := rowing_speed - current_speed
  let downstream_speed := rowing_speed + current_speed
  (total_time * upstream_speed * downstream_speed) / (2 * (upstream_speed + downstream_speed))

/-- Proves that the distance to the place is 96 km given the specified conditions -/
theorem distance_to_place (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) 
    (h1 : rowing_speed = 10)
    (h2 : current_speed = 2)
    (h3 : total_time = 20) :
  calculate_distance rowing_speed current_speed total_time = 96 := by
  sorry

/-- Proves that the distance remains 96 km even with additional obstacle navigation time -/
theorem distance_with_obstacles (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ)
    (obstacle_time_upstream : ℝ) (obstacle_time_downstream : ℝ)
    (h1 : rowing_speed = 10)
    (h2 : current_speed = 2)
    (h3 : total_time = 20)
    (h4 : obstacle_time_upstream = 1)
    (h5 : obstacle_time_downstream = 2) :
  calculate_distance rowing_speed current_speed total_time = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_place_distance_with_obstacles_l826_82601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion1_better_for_450_equal_payment_price_promotion2_better_range_l826_82664

-- Define the discount functions
def discount1 (price : ℝ) : ℝ := 0.8 * price

noncomputable def discount2 (price : ℝ) : ℝ :=
  price - (⌊price / 300⌋ * 80)

-- Theorem 1: Promotion 1 is more cost-effective for a $450 item
theorem promotion1_better_for_450 :
  discount1 450 < discount2 450 := by sorry

-- Theorem 2: The price where both promotions result in the same payment
theorem equal_payment_price :
  ∃ (x : ℝ), 300 ≤ x ∧ x < 500 ∧ discount1 x = discount2 x ∧ x = 400 := by sorry

-- Theorem 3: Range where Promotion 2 is more cost-effective
theorem promotion2_better_range (a : ℝ) :
  (300 ≤ a ∧ a < 400) ∨ (600 ≤ a ∧ a < 800) ↔ discount2 a < discount1 a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion1_better_for_450_equal_payment_price_promotion2_better_range_l826_82664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_dividing_f_seq_l826_82632

-- Define the polynomial f(x) = x^1998 - x^199 + x^19 + 1
def f (x : ℤ) : ℤ := x^1998 - x^199 + x^19 + 1

-- Define the sequence of f(n) for positive integers n
def f_seq : ℕ → ℤ
  | n => f (n : ℤ)

-- Statement of the theorem
theorem infinite_primes_dividing_f_seq :
  ∃ (S : Set ℕ), Set.Infinite S ∧ (∀ p ∈ S, Nat.Prime p) ∧
    (∀ p ∈ S, ∃ n : ℕ, (p : ℤ) ∣ f_seq n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_primes_dividing_f_seq_l826_82632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_of_vectors_l826_82649

theorem max_dot_product_of_vectors (a b c : ℝ × ℝ) : 
  (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 1) → 
  (Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 1) → 
  (Real.sqrt ((c.1 ^ 2) + (c.2 ^ 2)) = 1) → 
  (a.1 * b.1 + a.2 * b.2 = 1/2) →
  (∃ (x : ℝ), ((a.1 - b.1) * (a.1 - 2*c.1) + (a.2 - b.2) * (a.2 - 2*c.2) ≤ x) ∧ 
   ∀ (y : ℝ), ((a.1 - b.1) * (a.1 - 2*c.1) + (a.2 - b.2) * (a.2 - 2*c.2) ≤ y → x ≤ y)) →
  x = 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_dot_product_of_vectors_l826_82649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_truthful_dwarfs_l826_82608

/-- Represents the ice cream preferences of dwarfs -/
inductive IceCream
| Vanilla
| Chocolate
| Fruit
deriving DecidableEq

/-- Represents a dwarf -/
structure Dwarf where
  truthful : Bool
  favorite : IceCream
deriving DecidableEq

theorem four_truthful_dwarfs 
  (dwarfs : Finset Dwarf)
  (total_count : dwarfs.card = 10)
  (truth_or_lie : ∀ d ∈ dwarfs, d.truthful ∨ ¬d.truthful)
  (one_favorite : ∀ d ∈ dwarfs, 
    (d.favorite = IceCream.Vanilla) ∨ 
    (d.favorite = IceCream.Chocolate) ∨ 
    (d.favorite = IceCream.Fruit))
  (all_vanilla : dwarfs.card = (dwarfs.filter (λ d => 
    (d.truthful ∧ d.favorite = IceCream.Vanilla) ∨ 
    (¬d.truthful ∧ d.favorite ≠ IceCream.Vanilla))).card)
  (half_chocolate : (dwarfs.card / 2) = (dwarfs.filter (λ d => 
    (d.truthful ∧ d.favorite = IceCream.Chocolate) ∨ 
    (¬d.truthful ∧ d.favorite ≠ IceCream.Chocolate))).card)
  (one_fruit : 1 = (dwarfs.filter (λ d => 
    (d.truthful ∧ d.favorite = IceCream.Fruit) ∨ 
    (¬d.truthful ∧ d.favorite ≠ IceCream.Fruit))).card)
  : (dwarfs.filter (λ d => d.truthful)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_truthful_dwarfs_l826_82608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l826_82668

theorem sqrt_calculations : 
  (2 * Real.sqrt 27 - 3 * Real.sqrt 3 = 3 * Real.sqrt 3) ∧
  (2 * Real.sqrt 18 * 4 * Real.sqrt (1/2) + 3 * Real.sqrt 12 / Real.sqrt 3 = 30) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l826_82668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l826_82646

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Sum of distances from a point to two other points -/
noncomputable def sum_distances (p a b : Point) : ℝ :=
  distance p a + distance p b

/-- The point A -/
def A : Point := ⟨1, 3⟩

/-- The point B -/
def B : Point := ⟨5, 1⟩

/-- Theorem: The point (4,0) minimizes the sum of distances to A and B on the x-axis -/
theorem min_sum_distances :
  ∀ p : Point, p.y = 0 → sum_distances ⟨4, 0⟩ A B ≤ sum_distances p A B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l826_82646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l826_82638

/-- The angle in degrees that the hour hand moves per hour -/
noncomputable def hour_hand_speed : ℝ := 30

/-- The angle in degrees that the minute hand moves per minute -/
noncomputable def minute_hand_speed : ℝ := 6

/-- The time in hours since the start of the clock (12:00) -/
noncomputable def hours : ℝ := 3 + 40 / 60

/-- The position of the hour hand at the given time -/
noncomputable def hour_hand_position : ℝ := hour_hand_speed * hours

/-- The position of the minute hand at the given time -/
noncomputable def minute_hand_position : ℝ := minute_hand_speed * (hours * 60 % 60)

/-- The angle between the hour hand and minute hand -/
noncomputable def clock_angle : ℝ := abs (minute_hand_position - hour_hand_position)

theorem clock_angle_at_3_40 : clock_angle = 130 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l826_82638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_13_l826_82631

theorem sum_remainder_mod_13 (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_13_l826_82631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_work_hours_l826_82680

/-- Calculates the pay for a worker given their base rate, regular hours, overtime rate, and total hours worked. -/
def calculate_pay (base_rate : ℚ) (regular_hours : ℚ) (overtime_rate : ℚ) (total_hours : ℚ) : ℚ :=
  if total_hours ≤ regular_hours then
    base_rate * total_hours
  else
    base_rate * regular_hours + overtime_rate * (total_hours - regular_hours)

/-- Proves that Harry worked 34 hours given the payment conditions and James' work hours. -/
theorem harry_work_hours (x : ℚ) (h_pos : x > 0) : 
  calculate_pay x 18 (1.5 * x) 34 = calculate_pay x 40 (2 * x) 41 := by
  sorry

#check harry_work_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_work_hours_l826_82680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_efficiency_ratio_l826_82642

/-- The work ratios of projects A, B, and C -/
def work_ratios : Fin 3 → ℝ
| 0 => 1
| 1 => 2
| 2 => 3

/-- The work efficiency of each team -/
noncomputable def efficiency : Fin 3 → ℝ := sorry

/-- The amount of work completed by each team after k days -/
noncomputable def work_completed (k : ℝ) (i : Fin 3) : ℝ := k / efficiency i

/-- The amount of work not completed by each team after k days -/
noncomputable def work_remaining (k : ℝ) (i : Fin 3) : ℝ := work_ratios i - work_completed k i

/-- The conditions given in the problem -/
def problem_conditions (k : ℝ) : Prop :=
  work_completed k 0 = (1/2) * work_remaining k 1 ∧
  work_completed k 1 = (1/3) * work_remaining k 2 ∧
  work_completed k 2 = work_remaining k 0

/-- The theorem to be proved -/
theorem work_efficiency_ratio (k : ℝ) (hk : k > 0) :
  problem_conditions k →
  ∃ (c : ℝ), c > 0 ∧ 
    efficiency 0 = (4/c) ∧ 
    efficiency 1 = (6/c) ∧ 
    efficiency 2 = (3/c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_efficiency_ratio_l826_82642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_is_six_elevenths_l826_82623

/-- The set of ordered pairs (x, y) of non-negative integers satisfying x + y ≤ 10 -/
def S : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 + p.2 ≤ 10) (Finset.product (Finset.range 11) (Finset.range 11))

/-- The subset of S where the sum x + y is even -/
def E : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => Even (p.1 + p.2)) S

/-- The probability of selecting an element from E given a uniform distribution over S -/
def prob_even_sum : ℚ :=
  (E.card : ℚ) / (S.card : ℚ)

theorem prob_even_sum_is_six_elevenths : prob_even_sum = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_is_six_elevenths_l826_82623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_all_reals_l826_82616

/-- The function f(x) with parameter k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (3 * x^2 + 4 * x - 7) / (-7 * x^2 + 4 * x + k)

/-- The domain of f(x) is all real numbers iff k < -4/7 -/
theorem domain_of_f_is_all_reals (k : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f k x = y) ↔ k < -4/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_all_reals_l826_82616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l826_82610

theorem quadratic_factorization (b : ℤ) :
  (∃ (m n p q : ℤ), (15 : ℤ) * x^2 + b * x + 15 = (m * x + n) * (p * x + q) ∧ 
   (Nat.Prime m.natAbs ∨ Nat.Prime n.natAbs) ∧ 
   (Nat.Prime p.natAbs ∨ Nat.Prime q.natAbs)) →
  ∃ (k : ℤ), b = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l826_82610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parameter_values_l826_82698

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-3/5 * t + 2, 4/5 * t)

/-- Circle C in polar form -/
noncomputable def circle_C (a θ : ℝ) : ℝ := a * Real.sin θ

/-- Chord length condition -/
def chord_condition (a : ℝ) : Prop :=
  let d := |3*a/2 - 8| / 5
  d = Real.sqrt 3 * (a/2)

/-- Theorem: Given the line l and circle C with the chord condition, 
    the parameter a of the circle must be either 32 or 32/11 -/
theorem circle_parameter_values :
  ∃ (a : ℝ), (chord_condition a) ∧ (a = 32 ∨ a = 32/11) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parameter_values_l826_82698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_theorem_l826_82625

/-- The height of a building with specified floor heights -/
def buildingHeight (totalFloors : ℕ) (standardHeight : ℝ) (specialHeight : ℝ) (specialFloors : ℕ) : ℝ :=
  (standardHeight * (totalFloors - specialFloors : ℝ)) + (specialHeight * (specialFloors : ℝ))

/-- Theorem: The height of a 20-floor building with specific floor heights is 61 meters -/
theorem building_height_theorem :
  buildingHeight 20 3 3.5 2 = 61 := by
  -- Unfold the definition of buildingHeight
  unfold buildingHeight
  -- Simplify the arithmetic
  simp [Nat.cast_sub, Nat.cast_two]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_theorem_l826_82625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l826_82693

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point is on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Configuration of a circle and three lines in a plane -/
structure Configuration where
  circle : Circle
  line1 : Line
  line2 : Line
  line3 : Line
  distinct_lines : line1 ≠ line2 ∧ line1 ≠ line3 ∧ line2 ≠ line3
  not_parallel : ¬ (Line.parallel line1 line2) ∧ 
                 ¬ (Line.parallel line1 line3) ∧ 
                 ¬ (Line.parallel line2 line3)
  not_all_intersect : ¬ (∃ p, Line.contains line1 p ∧ 
                              Line.contains line2 p ∧ 
                              Line.contains line3 p)

/-- The number of intersection points in a configuration -/
def num_intersections (config : Configuration) : ℕ :=
  sorry

/-- Theorem stating the maximum number of intersections -/
theorem max_intersections (config : Configuration) :
  num_intersections config ≤ 9 ∧ 
  ∃ (config' : Configuration), num_intersections config' = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersections_l826_82693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_symmetry_l826_82672

-- Define the original line
def original_line (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := 3*x + 4*y + 5 = 0

-- Define a point on the original line
def point_on_original (x y : ℝ) : Prop := original_line x y

-- Define the reflection of a point across the x-axis
def reflect_across_x_axis (x y : ℝ) : ℝ × ℝ := (x, -y)

-- Theorem statement
theorem line_symmetry :
  ∀ x y : ℝ, point_on_original x y →
    (let (x', y') := reflect_across_x_axis x y; symmetric_line x' y') :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_symmetry_l826_82672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l826_82620

/-- Curve C1 in parametric form -/
noncomputable def C1 (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)

/-- Curve C2 in polar form -/
noncomputable def C2 (θ : ℝ) : ℝ := 8 * Real.sin θ

/-- The angle of the ray -/
noncomputable def θ₀ : ℝ := 2 * Real.pi / 3

/-- Point A: intersection of ray and C1 -/
noncomputable def A : ℝ × ℝ := C1 (Real.arcsin (Real.sin θ₀ / 2))

/-- Point B: intersection of ray and C2 -/
noncomputable def B : ℝ × ℝ := (C2 θ₀ * Real.cos θ₀, C2 θ₀ * Real.sin θ₀)

/-- The distance between points A and B is 2√3 -/
theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l826_82620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_l826_82605

noncomputable section

variable (f : ℝ → ℝ)

axiom f_decreasing : ∀ x y, x < y → f x > f y

axiom f_derivative_condition : ∀ x, f x / (deriv f x) < 1 - x

theorem f_positive : ∀ x, f x > 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_l826_82605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_factors_of_given_number_l826_82652

/-- The number of distinct natural-number factors of 8^2 * 9^3 * 5^5 * 7^1 -/
def num_factors : ℕ := 588

/-- The given number -/
def given_number : ℕ := 8^2 * 9^3 * 5^5 * 7^1

theorem num_factors_of_given_number :
  (Finset.filter (· ∣ given_number) (Finset.range (given_number + 1))).card = num_factors := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_factors_of_given_number_l826_82652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_three_l826_82665

/-- Given vectors a and b in ℝ², prove that the projection of a onto the unit vector in the direction of b is 3. -/
theorem projection_equals_three (a b : ℝ × ℝ) : 
  a = (0, 2 * Real.sqrt 3) → 
  b = (1, Real.sqrt 3) → 
  let e : ℝ × ℝ := (1 / Real.sqrt (b.1^2 + b.2^2)) • b
  (a.1 * e.1 + a.2 * e.2 : ℝ) = 3 := by
  intro ha hb
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_three_l826_82665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_two_common_tangents_l826_82612

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0

-- Define the center and radius of each circle
def center1 : ℝ × ℝ := (1, 0)
def radius1 : ℝ := 1
def center2 : ℝ × ℝ := (0, -2)
def radius2 : ℝ := 2

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Define a function to represent the number of common tangents
def number_of_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem circles_have_two_common_tangents :
  (distance_between_centers > radius1 + radius2 - 1) ∧
  (distance_between_centers < radius1 + radius2 + 1) →
  (∃ (n : ℕ), n = 2 ∧ n = number_of_common_tangents circle1 circle2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_have_two_common_tangents_l826_82612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l826_82695

/-- The value of m for a hyperbola with equation x²/16 - y²/m = 1 and focal length 4√5 -/
theorem hyperbola_m_value (m : ℝ) : 
  (∀ x y : ℝ, x^2/16 - y^2/m = 1) →  -- Hyperbola equation
  (2 * Real.sqrt (16 + m) = 4 * Real.sqrt 5) →  -- Focal length condition
  m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_m_value_l826_82695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_square_l826_82688

/-- A token in the plane -/
structure Token where
  x : ℝ
  y : ℝ

/-- A rectangle defined by four tokens -/
structure Rectangle where
  A : Token
  B : Token
  C : Token
  D : Token

/-- Predicate to check if three tokens are collinear -/
def collinear (t1 t2 t3 : Token) : Prop :=
  (t2.x - t1.x) * (t3.y - t1.y) = (t3.x - t1.x) * (t2.y - t1.y)

/-- Predicate to check if a rectangle is a square -/
def is_square (r : Rectangle) : Prop :=
  (r.A.x - r.B.x)^2 + (r.A.y - r.B.y)^2 = (r.B.x - r.C.x)^2 + (r.B.y - r.C.y)^2

/-- Predicate to check if two rectangles are similar but not equal -/
def similar_not_equal (r1 r2 : Rectangle) : Prop :=
  ∃ k : ℝ, k ≠ 1 ∧
    (r2.A.x - r2.B.x)^2 + (r2.A.y - r2.B.y)^2 = k^2 * ((r1.A.x - r1.B.x)^2 + (r1.A.y - r1.B.y)^2) ∧
    (r2.B.x - r2.C.x)^2 + (r2.B.y - r2.C.y)^2 = k^2 * ((r1.B.x - r1.C.x)^2 + (r1.B.y - r1.C.y)^2)

/-- The main theorem -/
theorem rectangle_to_square (pi pi' : Rectangle) :
  (∀ t1 t2 t3 : Token, ¬collinear t1 t2 t3) →
  similar_not_equal pi pi' →
  is_square pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_square_l826_82688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_l826_82651

noncomputable def arithmetic_mean (s : Finset ℝ) : ℝ := (s.sum id) / s.card

theorem arithmetic_mean_problem (x y : ℝ) :
  let s : Finset ℝ := {8, y, 24, 6, x}
  arithmetic_mean s = 12 ∧ x = 2 * y → x = 44/3 ∧ y = 22/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_l826_82651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_contribution_l826_82606

/-- Represents the capital contribution and duration for a business partner -/
structure Partner where
  contribution : ℚ
  duration : ℚ

/-- Represents a business partnership between two partners -/
structure Partnership where
  a : Partner
  b : Partner
  total_capital : ℚ

/-- The fraction of profit received by partner b -/
def b_profit_fraction : ℚ := 2/3

/-- Theorem stating the fraction of capital contributed by partner a -/
theorem partner_a_contribution (p : Partnership) 
  (h1 : p.a.duration = 15)
  (h2 : p.b.duration = 10)
  (h3 : p.a.contribution + p.b.contribution = p.total_capital)
  (h4 : p.a.contribution * p.a.duration / (p.b.contribution * p.b.duration) = (1 - b_profit_fraction) / b_profit_fraction) :
  p.a.contribution / p.total_capital = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_contribution_l826_82606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l826_82686

/-- Calculates the length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 450)
  (h2 : time_platform = 39)
  (h3 : time_pole = 18) :
  (train_length / time_pole) * time_platform - train_length = 525 := by
  -- Define train speed
  let train_speed := train_length / time_pole
  -- Define platform length
  let platform_length := train_speed * time_platform - train_length
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l826_82686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lopish_word_count_l826_82691

/-- The number of letters in the extended Lopish alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length in the Lopish language -/
def max_word_length : ℕ := 5

/-- Calculate the number of words of a given length that contain at least one 'B' -/
def words_with_b (length : ℕ) : ℕ :=
  alphabet_size^length - (alphabet_size - 1)^length

/-- The total number of valid words in the Lopish language -/
def total_valid_words : ℕ :=
  (List.range max_word_length).map (λ l ↦ words_with_b (l + 1)) |>.sum

/-- Theorem stating the total number of valid words in the Lopish language -/
theorem lopish_word_count : total_valid_words = 1855701 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lopish_word_count_l826_82691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_decreasing_interval_subset_domain_l826_82644

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 5*x - 6)

-- State the theorem
theorem decreasing_interval_of_f :
  ∀ x ∈ Set.Ioo (5/2 : ℝ) 3,
    ∀ y ∈ Set.Ioo (5/2 : ℝ) 3,
      x < y → f x > f y :=
by sorry

-- Define the domain of f
def domain_of_f : Set ℝ := Set.Ioo 2 3

-- State that the decreasing interval is a subset of the domain
theorem decreasing_interval_subset_domain :
  Set.Ioo (5/2 : ℝ) 3 ⊆ domain_of_f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_decreasing_interval_subset_domain_l826_82644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l826_82657

def my_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < 0) ∧ 
  (∀ n, 2 * a n = 3 * a (n + 1)) ∧
  (a 2 * a 5 = 8 / 27)

theorem sequence_formula (a : ℕ → ℝ) (h : my_sequence a) :
  ∃ C : ℝ, ∀ n : ℕ, a n = C * (2/3)^n ∧ C = -(3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l826_82657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_in_rectangular_prism_l826_82679

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Check if a point is on a line segment between two other points -/
def isOnLineSegment (p q r : Point3D) : Prop :=
  distance p q + distance q r = distance p r

/-- Check if a point is on the base of the rectangular prism -/
def isOnBase (prism : RectangularPrism) (p : Point3D) : Prop :=
  p.z = prism.A.z ∧ 
  prism.A.x ≤ p.x ∧ p.x ≤ prism.C.x ∧
  prism.A.y ≤ p.y ∧ p.y ≤ prism.C.y

/-- The main theorem -/
theorem min_distance_in_rectangular_prism (prism : RectangularPrism) 
  (hAB : distance prism.A prism.B = 2 * Real.sqrt 2)
  (hBC : distance prism.B prism.C = 2)
  (hAA₁ : distance prism.A prism.A₁ = 2) :
  ∃ (P Q : Point3D), 
    isOnLineSegment prism.A prism.C₁ P ∧
    isOnBase prism Q ∧
    ∀ (P' Q' : Point3D), 
      isOnLineSegment prism.A prism.C₁ P' → 
      isOnBase prism Q' →
      distance prism.B₁ P + distance P Q ≤ distance prism.B₁ P' + distance P' Q' ∧
      distance prism.B₁ P + distance P Q = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_in_rectangular_prism_l826_82679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_to_base_ratio_correct_l826_82678

/-- Regular triangular pyramid with base side length a and square cross-section of area m² -/
structure RegularTriangularPyramid where
  a : ℝ
  m : ℝ
  a_pos : 0 < a
  m_pos : 0 < m
  m_lt_a : m < a

/-- The ratio of lateral surface area to base area for a regular triangular pyramid -/
noncomputable def lateral_to_base_ratio (p : RegularTriangularPyramid) : ℝ :=
  Real.sqrt (9 * p.m^2 - 3 * p.a^2 + 6 * p.a * p.m) / (p.a - p.m)

theorem lateral_to_base_ratio_correct (p : RegularTriangularPyramid) :
  lateral_to_base_ratio p = (3 * p.a^2 * Real.sqrt (3 * p.m^2 - p.a^2 + 2 * p.a * p.m)) /
    (4 * (p.a - p.m)) / (p.a^2 * Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_to_base_ratio_correct_l826_82678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_leq_half_range_l826_82669

theorem cos_leq_half_range (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.cos x ≤ 1/2 ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_leq_half_range_l826_82669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_steps_from_neg_ten_to_thirty_l826_82690

/-- Represents an equally spaced number line -/
structure EquallySpacedNumberLine where
  start : ℝ
  stop : ℝ
  num_steps : ℕ

/-- Calculates the position on the number line after a given number of steps -/
noncomputable def position_after_steps (line : EquallySpacedNumberLine) (steps : ℕ) : ℝ :=
  line.start + (line.stop - line.start) * (steps : ℝ) / (line.num_steps : ℝ)

theorem five_steps_from_neg_ten_to_thirty (line : EquallySpacedNumberLine) 
  (h1 : line.start = -10)
  (h2 : line.stop = 30)
  (h3 : line.num_steps = 8) :
  position_after_steps line 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_steps_from_neg_ten_to_thirty_l826_82690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_speed_calculation_l826_82615

/-- Calculates the speed of the second part of a trip given the total distance,
    average speed, first part distance, and first part speed. -/
noncomputable def second_part_speed (total_distance : ℝ) (average_speed : ℝ) 
                      (first_part_distance : ℝ) (first_part_speed : ℝ) : ℝ :=
  let total_time := total_distance / average_speed
  let first_part_time := first_part_distance / first_part_speed
  let second_part_time := total_time - first_part_time
  let second_part_distance := total_distance - first_part_distance
  second_part_distance / second_part_time

/-- Theorem stating that for a 60 km trip with 40 km/h average speed,
    where the first 30 km is traveled at 60 km/h, 
    the speed for the remaining 30 km must be 30 km/h. -/
theorem trip_speed_calculation :
  second_part_speed 60 40 30 60 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_speed_calculation_l826_82615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_order_l826_82641

-- Define the reading amounts as real numbers
variable (A B C D : ℝ)

-- Define the conditions
def condition1 (A B C D : ℝ) : Prop := A + C = B + D
def condition2 (A B C D : ℝ) : Prop := A + B > C + D
def condition3 (A B C D : ℝ) : Prop := D > B + C

-- Theorem to prove
theorem reading_order (h1 : condition1 A B C D) (h2 : condition2 A B C D) (h3 : condition3 A B C D) :
  A > D ∧ D > B ∧ B > C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_order_l826_82641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_beat_distance_l826_82647

/-- The distance by which runner A beats runner B in a race -/
noncomputable def beat_distance (race_distance : ℝ) (a_time : ℝ) (time_difference : ℝ) : ℝ :=
  (race_distance / a_time) * time_difference

theorem race_beat_distance :
  let race_distance : ℝ := 1000
  let a_time : ℝ := 156.67
  let time_difference : ℝ := 10
  abs (beat_distance race_distance a_time time_difference - 63.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_beat_distance_l826_82647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_6_l826_82658

-- Define a fair 6-sided die
def fairDie : Finset ℕ := Finset.range 6

-- Define the sample space of rolling two dice
def sampleSpace : Finset (ℕ × ℕ) := fairDie.product fairDie

-- Define the event of at least one 6 appearing
def atLeastOne6 : Set (ℕ × ℕ) := {p | p.1 = 5 ∨ p.2 = 5}

-- Statement to prove
theorem prob_at_least_one_6 : 
  (sampleSpace.filter (λ p => p.1 = 5 ∨ p.2 = 5)).card / sampleSpace.card = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_6_l826_82658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_area_ratio_theorem_l826_82645

/-- The ratio of the cross-section area to the total surface area of a cone --/
noncomputable def cone_area_ratio (α β : Real) : Real :=
  Real.sin α / (4 * Real.pi * Real.cos β * (Real.cos (β / 2))^2)

/-- Theorem stating the ratio of the cross-section area to the total surface area of a cone --/
theorem cone_area_ratio_theorem (α β : Real) 
  (hα : 0 < α ∧ α < Real.pi) 
  (hβ : 0 < β ∧ β < Real.pi/2) :
  let S_sec := (1/2) * Real.pi * Real.sin α  -- Area of the cross-section
  let S_total := Real.pi * (Real.cos β + Real.cos β ^ 2)  -- Total surface area of the cone
  S_sec / S_total = cone_area_ratio α β := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_area_ratio_theorem_l826_82645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l826_82675

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Theorem: The eccentricity of the ellipse x^2/9 + y^2/4 = 1 is √5/3 -/
theorem ellipse_eccentricity :
  let a : ℝ := 3
  let b : ℝ := 2
  eccentricity a b = Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l826_82675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l826_82673

open Real

-- Define the function
noncomputable def f (x : ℝ) := 2 * sin (π / 6 - 2 * x)

-- State the theorem
theorem f_increasing_interval :
  ∀ x ∈ Set.Icc (π / 3) (5 * π / 6),
    x ∈ Set.Icc 0 π →
    ∀ y ∈ Set.Icc (π / 3) (5 * π / 6),
      x < y → f x < f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l826_82673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_peach_pies_l826_82637

-- Define the problem parameters
def apple_pies : ℕ := 4
def blueberry_pies : ℕ := 3
def fruit_per_pie : ℕ := 3
def apple_blueberry_price : ℚ := 1
def peach_price : ℚ := 2
def total_spent : ℚ := 51

-- Define the function to calculate the number of peach pies
def peach_pies : ℕ :=
  let apple_blueberry_cost := (apple_pies + blueberry_pies) * fruit_per_pie * apple_blueberry_price
  let peach_cost := total_spent - apple_blueberry_cost
  let peach_pie_cost := fruit_per_pie * peach_price
  (peach_cost / peach_pie_cost).floor.toNat

-- State the theorem
theorem michael_peach_pies : peach_pies = 5 := by
  -- Unfold the definition of peach_pies
  unfold peach_pies
  -- Simplify the arithmetic expressions
  simp [apple_pies, blueberry_pies, fruit_per_pie, apple_blueberry_price, peach_price, total_spent]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_peach_pies_l826_82637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l826_82604

theorem inequality_proof (φ : Real) (h : φ ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l826_82604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_difference_l826_82682

theorem triangle_side_difference : ℕ := by
  -- Define the triangle sides
  let side1 : ℕ := 7
  let side2 : ℕ := 10

  -- Define the range of possible y values
  let y_min : ℕ := 4
  let y_max : ℕ := 16

  -- Define the triangle inequality conditions
  let triangle_inequality (y : ℕ) : Prop :=
    y + side1 > side2 ∧ y + side2 > side1 ∧ side1 + side2 > y

  -- Define the condition that y is within the valid range
  let y_in_range (y : ℕ) : Prop :=
    y ≥ y_min ∧ y ≤ y_max

  -- The theorem statement
  have side_difference_is_12 :
    ∀ y : ℕ, triangle_inequality y → y_in_range y →
    (y_max - y_min : ℕ) = 12 := by
    sorry

  exact 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_difference_l826_82682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_general_equation_l826_82634

-- Define the curve C
noncomputable def C (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt t - 1 / Real.sqrt t, 3 * (t + 1 / t) + 2)

-- Theorem statement
theorem curve_C_general_equation :
  ∀ (t : ℝ), t > 0 →
    let (x, y) := C t
    x^2 = (y - 8) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_general_equation_l826_82634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l826_82603

/-- Two plane vectors are collinear if and only if there exist non-zero real numbers
    that make their linear combination equal to the zero vector. -/
theorem vector_collinearity (a b : ℝ × ℝ) :
  (∃ (k : ℝ), b = k • a ∨ a = k • b) ↔
  (∃ (l1 l2 : ℝ), l1 ≠ 0 ∧ l2 ≠ 0 ∧ l1 • a + l2 • b = (0, 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l826_82603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_equality_l826_82643

theorem contrapositive_sine_equality (x y : ℝ) : 
  (¬(Real.sin x = Real.sin y) → ¬(x = y)) ↔ true :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_equality_l826_82643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_well_defined_iff_l826_82656

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (1 - a * x)

-- State the theorem
theorem f_well_defined_iff (a : ℝ) :
  (∀ x ≥ -1, 1 - a * x ≥ 0) ↔ a ∈ Set.Icc (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_well_defined_iff_l826_82656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l826_82671

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  first_positive : a 1 > 0
  diff_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : sum_n seq 5 = sum_n seq 9) : 
  (∀ n, n ≠ 7 → sum_n seq n ≤ sum_n seq 7) ∧ 
  sum_n seq 14 = 0 ∧
  (∀ n, n > 13 → sum_n seq n ≤ 0) ∧
  (∃ n, n = 13 ∧ sum_n seq n > 0) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l826_82671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_papi_calot_garden_solution_l826_82677

/-- Represents Papi Calot's garden planning problem -/
structure GardenProblem where
  rows : ℕ
  plants_per_row : ℕ
  total_plants : ℕ

/-- Calculates the number of additional plants needed -/
def additional_plants (g : GardenProblem) : ℕ :=
  g.total_plants - (g.rows * g.plants_per_row)

/-- Theorem stating the solution to Papi Calot's garden problem -/
theorem papi_calot_garden_solution (g : GardenProblem)
  (h1 : g.rows = 7)
  (h2 : g.plants_per_row = 18)
  (h3 : g.total_plants = 141) :
  additional_plants g = 15 := by
  sorry

#eval additional_plants { rows := 7, plants_per_row := 18, total_plants := 141 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_papi_calot_garden_solution_l826_82677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l826_82660

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (q : ℝ) (h_q_ne_0 : q ≠ 0) :
  (geometric_sum a₁ q 2 = 2 * (geometric_sequence a₁ q 2) + 3) →
  (geometric_sum a₁ q 3 = 2 * (geometric_sequence a₁ q 3) + 3) →
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l826_82660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_100_eq_128_5714286_l826_82694

open BigOperators

def series_sum (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, (3 + 9 * (k + 1 : ℚ)) / (8 ^ (100 - k))

theorem series_sum_100_eq_128_5714286 :
  series_sum 100 = 128571428571429 / 1000000000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_100_eq_128_5714286_l826_82694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_div_D_equals_16_l826_82609

-- Define the series C
noncomputable def C : ℝ := ∑' n, if n % 4 ≠ 0 then ((-1) ^ ((n - 2) / 4)) / (4 * n - 2) ^ 2 else 0

-- Define the series D
noncomputable def D : ℝ := ∑' n, 1 / (4 * n) ^ 2

-- Theorem statement
theorem C_div_D_equals_16 : C / D = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_div_D_equals_16_l826_82609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_x_plus_3pi_over_2_l826_82663

theorem cos_neg_x_plus_3pi_over_2 (x : ℝ) 
  (h1 : Real.tan x = -12/5) 
  (h2 : π/2 < x ∧ x < π) : 
  Real.cos (-x + 3*π/2) = -12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_x_plus_3pi_over_2_l826_82663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coordinates_l826_82639

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- A point lies on a coordinate axis if either its x or y coordinate is 0 -/
def onCoordinateAxis (p : Point) : Prop :=
  p.x = 0 ∨ p.y = 0

theorem triangle_coordinates :
  let O : Point := ⟨0, 0⟩
  let B : Point := ⟨1, 2⟩
  ∀ A : Point,
    onCoordinateAxis A →
    triangleArea O A B = 2 →
    (A = ⟨2, 0⟩ ∨ A = ⟨-2, 0⟩ ∨ A = ⟨0, 4⟩ ∨ A = ⟨0, -4⟩) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coordinates_l826_82639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_percentage_is_twenty_l826_82681

/-- Represents the percentage of total amount spent on each category and the tax rates --/
structure ShoppingBreakdown where
  clothing_percent : ℚ
  other_percent : ℚ
  clothing_tax : ℚ
  other_tax : ℚ
  total_tax_percent : ℚ

/-- Calculates the percentage spent on food given a shopping breakdown --/
def food_percent (s : ShoppingBreakdown) : ℚ :=
  100 - s.clothing_percent - s.other_percent

/-- Calculates the total tax percentage given a shopping breakdown --/
def calculate_total_tax (s : ShoppingBreakdown) : ℚ :=
  (s.clothing_percent * s.clothing_tax + s.other_percent * s.other_tax) / 100

/-- Theorem stating that given the shopping conditions, the percentage spent on food is 20% --/
theorem food_percentage_is_twenty (s : ShoppingBreakdown) 
  (h1 : s.clothing_percent = 50)
  (h2 : s.other_percent = 30)
  (h3 : s.clothing_tax = 4)
  (h4 : s.other_tax = 8)
  (h5 : s.total_tax_percent = 4.4)
  (h6 : calculate_total_tax s = s.total_tax_percent) :
  food_percent s = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_percentage_is_twenty_l826_82681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l826_82617

/-- Calculates the final amount of an investment with compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) (frequency : ℕ) : ℝ :=
  principal * (1 + rate / (frequency : ℝ)) ^ ((frequency : ℝ) * (time : ℝ))

/-- Theorem stating that an investment of $6000 at 10% annual compound interest for 2 years, 
    compounded annually, will result in a final amount of $7260 -/
theorem investment_result : 
  let principal := (6000 : ℝ)
  let rate := (0.10 : ℝ)
  let time := 2
  let frequency := 1
  compound_interest principal rate time frequency = 7260 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l826_82617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_5_4850_rounded_l826_82648

theorem log_5_4850_rounded : ⌊Real.log 4850 / Real.log 5 + 0.5⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_5_4850_rounded_l826_82648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l826_82689

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6) * Real.sin (2 * x) - 1 / 4

theorem symmetry_center_of_f :
  ∃ (c : ℝ), (∀ (x : ℝ), f (c + x) = f (c - x)) ∧ c = 7 * Real.pi / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l826_82689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_is_945_l826_82676

/-- A polynomial with integer coefficients satisfying specific conditions -/
def P : Polynomial ℤ → Prop :=
  λ p => ∃ (a : ℕ), a > 0 ∧
    p.eval 1 = a ∧
    p.eval 3 = a ∧
    p.eval 5 = a ∧
    p.eval 7 = a ∧
    p.eval 2 = -a ∧
    p.eval 4 = -a ∧
    p.eval 6 = -a ∧
    p.eval 8 = -a ∧
    p.eval 0 = 0 ∧
    p.eval 9 = a

/-- The smallest possible value of a satisfying the conditions -/
def smallest_a : ℕ := 945

/-- Theorem stating that 945 is the smallest possible value of a -/
theorem smallest_a_is_945 : 
  ∀ (p : Polynomial ℤ), P p → 
    ∀ (a : ℕ), a > 0 → 
    (p.eval 1 = a ∧ 
     p.eval 3 = a ∧ 
     p.eval 5 = a ∧ 
     p.eval 7 = a ∧ 
     p.eval 2 = -↑a ∧ 
     p.eval 4 = -↑a ∧ 
     p.eval 6 = -↑a ∧ 
     p.eval 8 = -↑a ∧ 
     p.eval 0 = 0 ∧ 
     p.eval 9 = a) → 
    a ≥ smallest_a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_is_945_l826_82676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_heating_rate_l826_82666

/-- The rate at which water temperature increases per minute -/
noncomputable def temperature_increase_rate (
  initial_temp : ℚ)
  (boiling_temp : ℚ)
  (pasta_cooking_time : ℚ)
  (total_cooking_time : ℚ) : ℚ :=
  let temp_difference := boiling_temp - initial_temp
  let mixing_time := pasta_cooking_time / 3
  let heating_time := total_cooking_time - pasta_cooking_time - mixing_time
  temp_difference / heating_time

theorem water_heating_rate :
  temperature_increase_rate 41 212 12 73 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_heating_rate_l826_82666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_subset_l826_82627

theorem perfect_square_subset (numbers : Finset ℕ) 
  (h1 : numbers.card = 1986)
  (h2 : (numbers.prod id).factors.toFinset.card = 1985) :
  ∃ (subset : Finset ℕ), subset ⊆ numbers ∧ subset.Nonempty ∧ 
  ∃ (n : ℕ), subset.prod id = n^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_subset_l826_82627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_l826_82687

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by a point and a slope -/
structure Line where
  point : Point
  slope : ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem triangle_area_is_nine (P : Point) (line1 line2 : Line) (Q R : Point) : 
  P.x = 1 ∧ P.y = 6 ∧
  line1.point = P ∧ line1.slope = 1 ∧
  line2.point = P ∧ line2.slope = 2 ∧
  Q.y = 0 ∧ R.y = 0 ∧
  Q.x = -5 ∧ R.x = -2 →
  triangleArea (R.x - Q.x) P.y = 9 := by
  sorry

#check triangle_area_is_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_nine_l826_82687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l826_82619

noncomputable def curve (x : ℝ) : ℝ := 1 / x

theorem area_of_triangle (A B : ℝ × ℝ) :
  (A.1 > 0 ∧ A.2 > 0 ∧ B.1 > 0 ∧ B.2 > 0) →  -- A and B in first quadrant
  A.2 = curve A.1 →                         -- A on the curve
  B.2 = curve B.1 →                         -- B on the curve
  Real.cos (Real.pi / 3) = (A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2)) →  -- ∠AOB = 60°
  A.1^2 + A.2^2 = B.1^2 + B.2^2 →           -- OA = OB
  (1/2) * A.1 * B.2 - (1/2) * A.2 * B.1 = Real.sqrt 3 :=  -- Area of triangle OAB
by sorry

#check area_of_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l826_82619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_first_five_primes_l826_82640

theorem smallest_four_digit_divisible_by_first_five_primes :
  ∃ (n : ℕ), 
    (n ≥ 1000) ∧ 
    (n < 10000) ∧ 
    (∀ p : ℕ, p ∈ ({2, 3, 5, 7, 11} : Set ℕ) → n % p = 0) ∧
    (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ (∀ p : ℕ, p ∈ ({2, 3, 5, 7, 11} : Set ℕ) → m % p = 0) → m ≥ n) ∧
    n = 2310 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_divisible_by_first_five_primes_l826_82640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_theorem_l826_82692

/-- Represents the number of contacts between people in a group -/
def contacts (n : ℕ) : ℕ → ℕ
| 0 => 1
| k + 1 => 3 * contacts n k

/-- The theorem statement -/
theorem contact_theorem (n : ℕ) (h1 : n ≥ 4) :
  (∀ (k : ℕ), ∀ (S : Finset (Fin n)), S.card = n - 2 →
    ∀ (i : Fin n), i ∈ S →
      (S.filter (λ j ↦ j ≠ i)).card = contacts n k) ↔
  (n = 4 ∨ n = 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contact_theorem_l826_82692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_squared_eq_eight_l826_82628

/-- A function y of x parameterized by a and b -/
noncomputable def y (a b x : ℝ) : ℝ := (a * Real.cos x + b * Real.sin x) * Real.cos x

/-- The maximum value of y is 2 -/
axiom y_max (a b : ℝ) : ∃ x, y a b x ≤ 2 ∧ ∀ x', y a b x' ≤ y a b x

/-- The minimum value of y is -1 -/
axiom y_min (a b : ℝ) : ∃ x, y a b x ≥ -1 ∧ ∀ x', y a b x' ≥ y a b x

/-- The theorem to be proved -/
theorem ab_squared_eq_eight (a b : ℝ) : (a * b)^2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_squared_eq_eight_l826_82628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_reduction_l826_82655

/-- Calculates the final price and total percent reduction of a coat after a series of price reductions and taxes --/
theorem coat_price_reduction (initial_price : ℝ) 
  (reduction1 reduction2 reduction3 : ℝ) 
  (tax1 tax2 tax3 : ℝ) : 
  initial_price = 500 ∧ 
  reduction1 = 0.1 ∧ 
  reduction2 = 0.15 ∧ 
  reduction3 = 0.2 ∧ 
  tax1 = 0.05 ∧ 
  tax2 = 0.08 ∧ 
  tax3 = 0.06 → 
  ∃ (final_price total_reduction : ℝ),
    (abs (final_price - 367.824) < 0.001 ∧ 
     abs (total_reduction - 0.2644) < 0.0001) ∧
    (final_price = initial_price * 
      (1 - reduction1) * (1 + tax1) * 
      (1 - reduction2) * (1 + tax2) * 
      (1 - reduction3) * (1 + tax3)) ∧
    (total_reduction = (initial_price - final_price) / initial_price) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coat_price_reduction_l826_82655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l826_82633

-- Define the function f(x) = √(3x - 2)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 * x - 2)

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 3 / (2 * Real.sqrt (3 * x - 2))

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 3 * x - 2 * y - 1 = 0 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l826_82633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_g_l826_82699

noncomputable def f (x : ℝ) : ℝ := Real.cos (4 * x - Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)

theorem smallest_positive_period_of_g :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), g (x + T) = g x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), g (x + T') = g x) → T ≤ T') ∧
  T = Real.pi := by
  sorry

#check smallest_positive_period_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_g_l826_82699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_coloring_l826_82662

/-- Represents a move on the chessboard -/
structure Move where
  width : Nat
  height : Nat
  /-- Both sides must be odd or both even, and at least one side > 1 -/
  valid : (width % 2 = height % 2) ∧ (width > 1 ∨ height > 1)

/-- Represents the state of the chessboard -/
def Board (n : Nat) := Fin n → Fin n → Bool

/-- Applies a move to the board -/
def applyMove (b : Board n) (m : Move) : Board n :=
  sorry

/-- Checks if all squares on the board have the same color -/
def allSameColor (b : Board n) : Prop :=
  sorry

/-- Creates an initial chessboard with alternating colors -/
def initialBoard (n : Nat) : Board n :=
  fun i j => (i.val + j.val) % 2 = 0

/-- Main theorem: It's possible to make all squares the same color
    if and only if n = 1 or n ≥ 3 -/
theorem chessboard_coloring (n : Nat) :
  (∃ (moves : List Move), allSameColor (moves.foldl applyMove (initialBoard n))) ↔
  (n = 1 ∨ n ≥ 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_coloring_l826_82662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l826_82611

/-- A predicate to determine if two expressions are mathematically equivalent. -/
def IsEquivalentExpression (e1 e2 : ℝ) : Prop :=
  ∀ x, e1 = x ↔ e2 = x

/-- A predicate to determine if a quadratic radical expression can be simplified. -/
def CanBeSimplified (e : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, f e ≠ e ∧ IsEquivalentExpression e (f e)

/-- A predicate to determine if an expression is the simplest quadratic radical among a list of expressions. -/
def IsSimplestQuadraticRadical (e : ℝ) (l : List ℝ) : Prop :=
  e ∈ l ∧ ∀ x ∈ l, ¬ (CanBeSimplified x) → x = e

/-- Given four quadratic radical expressions, prove that the first one is the simplest. -/
theorem simplest_quadratic_radical (a b : ℝ) :
  let expr1 := Real.sqrt (a^2 + b^2) / 2
  let expr2 := Real.sqrt (2 * a^2 * b)
  let expr3 := 1 / Real.sqrt 6
  let expr4 := Real.sqrt (8 * a)
  IsSimplestQuadraticRadical expr1 [expr1, expr2, expr3, expr4] := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l826_82611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l826_82614

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle)
  (h1 : (cos t.A) / (1 + sin t.A) = (sin (2 * t.B)) / (1 + cos (2 * t.B)))
  (h2 : t.C = 2 * π / 3) :
  t.B = π / 6 ∧ 
  (∀ (x : Triangle), x.a^2 + x.b^2 ≥ (4 * Real.sqrt 2 - 5) * x.c^2) ∧
  (∃ (x : Triangle), x.a^2 + x.b^2 = (4 * Real.sqrt 2 - 5) * x.c^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l826_82614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_vertex_in_second_quadrant_l826_82607

/-- The x-coordinate of the vertex of a parabola y = 4x^2 - 4(a+1)x + a -/
noncomputable def vertex_x (a : ℝ) : ℝ := (a + 1) / 2

/-- The y-coordinate of the vertex of a parabola y = 4x^2 - 4(a+1)x + a -/
noncomputable def vertex_y (a : ℝ) : ℝ := -a^2 - a - 1

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem no_vertex_in_second_quadrant :
  ∀ a : ℝ, ¬(in_second_quadrant (vertex_x a) (vertex_y a)) :=
by
  intro a
  simp [in_second_quadrant, vertex_x, vertex_y]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_vertex_in_second_quadrant_l826_82607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roger_spending_l826_82659

/-- Represents the fraction of allowance spent on movie ticket and snack -/
noncomputable def fraction_spent (A m s : ℝ) : ℝ :=
  (m + s) / A

/-- The problem statement -/
theorem roger_spending (A : ℝ) (A_pos : A > 0) :
  ∃ (m s : ℝ),
    m = (1/4 : ℝ) * (A - 2*s) ∧
    s = (1/10 : ℝ) * (A - (1/2)*m) ∧
    fraction_spent A m s = 229/390 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roger_spending_l826_82659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_l826_82622

/-- A coloring of a 10 × 10 board. -/
def Coloring := Fin 10 → Fin 10 → ℕ

/-- The property that each row contains at most 5 colors. -/
def RowConstraint (c : Coloring) : Prop :=
  ∀ i : Fin 10, Finset.card (Finset.image (c i) (Finset.univ : Finset (Fin 10))) ≤ 5

/-- The property that each column contains at most 5 colors. -/
def ColumnConstraint (c : Coloring) : Prop :=
  ∀ j : Fin 10, Finset.card (Finset.image (fun i => c i j) (Finset.univ : Finset (Fin 10))) ≤ 5

/-- The number of distinct colors used in a coloring. -/
def NumColors (c : Coloring) : ℕ :=
  Finset.card (Finset.image (fun (i, j) => c i j) (Finset.univ : Finset (Fin 10 × Fin 10)))

/-- The main theorem: The maximum number of colors that can be used is 41. -/
theorem max_colors :
  (∃ (c : Coloring), RowConstraint c ∧ ColumnConstraint c ∧ NumColors c = 41) ∧
  (∀ (c : Coloring), RowConstraint c → ColumnConstraint c → NumColors c ≤ 41) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_l826_82622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_savings_l826_82630

/-- Thomas's savings problem -/
theorem thomas_savings
  (allowance hourly_wage hours_per_week car_price additional_needed : ℕ) : 
  allowance = 50 →
  hourly_wage = 9 →
  hours_per_week = 30 →
  car_price = 15000 →
  additional_needed = 2000 →
  ∃ (weekly_spending : ℕ), 
    weekly_spending = 35 ∧
    (allowance * 52 + hourly_wage * hours_per_week * 52 - car_price - additional_needed) / 104 = weekly_spending :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_savings_l826_82630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l826_82635

/-- Two circles are externally tangent -/
def ExternallyTangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A smaller circle is internally tangent to a larger circle -/
def InternallyTangent (r_small r_large : ℝ) : Prop := sorry

/-- A chord is a common external tangent to two smaller circles inside a larger circle -/
def IsCommonExternalTangent (chord r₁ r₂ r₃ : ℝ) : Prop := sorry

/-- Given three circles with radii 4, 8, and 12, where the two smaller circles
    are externally tangent to each other and internally tangent to the largest circle,
    the square of the length of the chord of the largest circle that is a common
    external tangent to the two smaller circles is equal to 3584/9. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : r₃ = 12)
  (h_ext_tangent : ExternallyTangent r₁ r₂)
  (h_int_tangent₁ : InternallyTangent r₁ r₃)
  (h_int_tangent₂ : InternallyTangent r₂ r₃)
  (chord : ℝ) (h_chord : IsCommonExternalTangent chord r₁ r₂ r₃) :
  chord^2 = 3584/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l826_82635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l826_82684

/-- A line with slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Calculate the y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Calculate the x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ :=
  -y_intercept l / l.slope

/-- Calculate the area of a triangle given base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  0.5 * base * height

/-- The main theorem -/
theorem area_of_quadrilateral_OBEC : 
  let line1 : Line := { slope := -3, point := (3, 3) }
  let A : ℝ × ℝ := (x_intercept line1, 0)
  let B : ℝ × ℝ := (0, y_intercept line1)
  let C : ℝ × ℝ := (6, 0)
  let E : ℝ × ℝ := (3, 3)
  let line2 : Line := { slope := (0 - E.2) / (6 - E.1), point := C }
  let D : ℝ × ℝ := (0, y_intercept line2)
  let area_OBE := triangle_area B.2 E.1
  let area_OEC := triangle_area C.1 E.2
  area_OBE + area_OEC = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_OBEC_l826_82684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l826_82697

theorem divisibility_theorem (p : ℕ) (a b α β : ℤ) (hp : Nat.Prime p) 
  (h1 : (p : ℤ) ∣ (a * α + b)) (h2 : (p : ℤ) ∣ (a * β + b)) (h3 : ¬((p : ℤ) ∣ (α - β))) :
  ((p : ℤ) ∣ a) ∧ ((p : ℤ) ∣ b) ∧ (∀ x : ℤ, (p : ℤ) ∣ (a * x + b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l826_82697
