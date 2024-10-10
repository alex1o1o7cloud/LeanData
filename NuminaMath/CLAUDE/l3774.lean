import Mathlib

namespace grape_jelly_beans_problem_l3774_377456

theorem grape_jelly_beans_problem (g c : ℕ) : 
  g = 3 * c →                   -- Initial ratio
  g - 15 = 5 * (c - 5) →        -- Final ratio after eating
  g = 15                        -- Conclusion: original number of grape jelly beans
  := by sorry

end grape_jelly_beans_problem_l3774_377456


namespace sum_of_sequence_l3774_377410

/-- Calculates the sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- The number of terms in the sequence -/
def num_terms (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

theorem sum_of_sequence : 
  let a₁ := 71  -- First term
  let aₙ := 361 -- Last term
  let d := 10   -- Common difference
  let n := num_terms a₁ aₙ d
  arithmetic_sum a₁ aₙ n = 6480 :=
by sorry

end sum_of_sequence_l3774_377410


namespace food_festival_total_cost_l3774_377419

def food_festival_cost (hot_dog_price1 hot_dog_price2 hot_dog_price3 : ℚ)
                       (ice_cream_price1 ice_cream_price2 : ℚ)
                       (lemonade_price1 lemonade_price2 lemonade_price3 : ℚ) : ℚ :=
  3 * hot_dog_price1 + 3 * hot_dog_price2 + 2 * hot_dog_price3 +
  2 * ice_cream_price1 + 3 * ice_cream_price2 +
  lemonade_price1 + lemonade_price2 + lemonade_price3

theorem food_festival_total_cost :
  food_festival_cost 0.60 0.75 0.90 1.50 2.00 2.50 3.00 3.50 = 23.85 := by
  sorry

end food_festival_total_cost_l3774_377419


namespace greatest_three_digit_number_l3774_377491

theorem greatest_three_digit_number : ∃ n : ℕ,
  n = 989 ∧
  n < 1000 ∧
  ∃ k : ℕ, n = 7 * k + 2 ∧
  ∃ m : ℕ, n = 4 * m + 1 ∧
  ∀ x : ℕ, x < 1000 → (∃ a : ℕ, x = 7 * a + 2) → (∃ b : ℕ, x = 4 * b + 1) → x ≤ n :=
by
  sorry

end greatest_three_digit_number_l3774_377491


namespace polynomial_coefficient_sums_l3774_377439

theorem polynomial_coefficient_sums :
  ∀ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (3 - 2 * x)^5 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) →
  (a₁ + a₂ + a₃ + a₄ + a₅ = -242) ∧
  (|a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 2882) := by
sorry

end polynomial_coefficient_sums_l3774_377439


namespace quadratic_roots_squared_l3774_377479

theorem quadratic_roots_squared (α β : ℝ) : 
  (α^2 - 3*α - 1 = 0) → 
  (β^2 - 3*β - 1 = 0) → 
  (α + β = 3) →
  (α * β = -1) →
  ((α^2)^2 - 11*(α^2) + 1 = 0) ∧ ((β^2)^2 - 11*(β^2) + 1 = 0) :=
by sorry

end quadratic_roots_squared_l3774_377479


namespace sin_value_for_given_condition_l3774_377457

theorem sin_value_for_given_condition (θ : Real) 
  (h1 : 5 * Real.tan θ = 2 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < π) : 
  Real.sin θ = (Real.sqrt 41 - 5) / 4 := by
  sorry

end sin_value_for_given_condition_l3774_377457


namespace negation_equivalence_l3774_377464

theorem negation_equivalence :
  (¬ ∃ a ∈ Set.Icc (0 : ℝ) 1, a^4 + a^2 > 1) ↔
  (∀ a ∈ Set.Icc (0 : ℝ) 1, a^4 + a^2 ≤ 1) := by
  sorry

end negation_equivalence_l3774_377464


namespace proposition_truth_l3774_377493

theorem proposition_truth (p q : Prop) (hp : p) (hq : ¬q) : (¬p) ∨ (¬q) := by
  sorry

end proposition_truth_l3774_377493


namespace parabola_point_movement_l3774_377429

/-- Represents a parabola of the form y = x^2 - 2mx - 3 -/
structure Parabola where
  m : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y = pt.x^2 - 2*p.m*pt.x - 3

/-- Calculates the vertex of the parabola -/
def vertex (p : Parabola) : Point :=
  { x := p.m, y := -(p.m^2) - 3 }

theorem parabola_point_movement (p : Parabola) (A : Point) (n b : ℝ) :
  on_parabola p { x := -2, y := n } →
  { x := 1, y := n - b } = vertex p →
  b = 9 := by sorry

end parabola_point_movement_l3774_377429


namespace yoongis_number_l3774_377471

theorem yoongis_number (x : ℤ) (h : x - 10 = 15) : x + 5 = 30 := by
  sorry

end yoongis_number_l3774_377471


namespace absolute_value_equation_product_l3774_377485

theorem absolute_value_equation_product (x₁ x₂ : ℝ) : 
  (|4 * x₁| + 3 = 35) ∧ (|4 * x₂| + 3 = 35) ∧ (x₁ ≠ x₂) → x₁ * x₂ = -64 :=
by sorry

end absolute_value_equation_product_l3774_377485


namespace three_can_volume_l3774_377476

theorem three_can_volume : 
  ∀ (v1 v2 v3 : ℕ),
  v2 = (3 * v1) / 2 →
  v3 = 64 * v1 / 3 →
  v1 + v2 + v3 < 30 →
  v1 + v2 + v3 = 23 :=
by
  sorry

end three_can_volume_l3774_377476


namespace a_less_than_b_plus_one_l3774_377418

theorem a_less_than_b_plus_one (a b : ℝ) (h : a < b) : a < b + 1 := by
  sorry

end a_less_than_b_plus_one_l3774_377418


namespace system_solvability_l3774_377402

/-- The set of values for parameter a such that the system has at least one solution -/
def ValidAValues : Set ℝ := {a | a < 0 ∨ a ≥ 2/3}

/-- The system of equations -/
def System (a b x y : ℝ) : Prop :=
  x = |y + a| + 4/a ∧ x^2 + y^2 + 24 + b*(2*y + b) = 10*x

/-- Theorem stating the condition for the existence of a solution -/
theorem system_solvability (a : ℝ) :
  (∃ b x y, System a b x y) ↔ a ∈ ValidAValues :=
sorry

end system_solvability_l3774_377402


namespace arithmetic_sequence_common_difference_l3774_377483

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def common_difference (a : ℕ → ℚ) : ℚ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_condition1 : a 7 - 2 * a 4 = -1)
  (h_condition2 : a 3 = 0) :
  common_difference a = -1/2 := by
  sorry

end arithmetic_sequence_common_difference_l3774_377483


namespace geometric_sequence_nth_term_l3774_377482

theorem geometric_sequence_nth_term (a₁ q : ℚ) (n : ℕ) (h1 : a₁ = 1/2) (h2 : q = 1/2) :
  a₁ * q^(n - 1) = 1/32 → n = 5 := by
  sorry

end geometric_sequence_nth_term_l3774_377482


namespace point_movement_on_number_line_l3774_377427

theorem point_movement_on_number_line (A : ℝ) (movement : ℝ) : 
  A = -2 → movement = 4 → (A - movement = -6 ∨ A + movement = 2) := by
  sorry

end point_movement_on_number_line_l3774_377427


namespace min_top_supervisors_bound_l3774_377451

/-- Represents the structure of a company --/
structure Company where
  total_employees : ℕ
  supervisor_subordinate_sum : ℕ
  propagation_days : ℕ

/-- Calculates the minimum number of top-level supervisors --/
def min_top_supervisors (c : Company) : ℕ :=
  ((c.total_employees - 1) / (1 + c.supervisor_subordinate_sum + c.supervisor_subordinate_sum ^ 2 + c.supervisor_subordinate_sum ^ 3 + c.supervisor_subordinate_sum ^ 4)) + 1

/-- The theorem to be proved --/
theorem min_top_supervisors_bound (c : Company) 
  (h1 : c.total_employees = 50000)
  (h2 : c.supervisor_subordinate_sum = 7)
  (h3 : c.propagation_days = 4) :
  min_top_supervisors c ≥ 97 := by
  sorry

#eval min_top_supervisors ⟨50000, 7, 4⟩

end min_top_supervisors_bound_l3774_377451


namespace shooting_competition_problem_prove_shooting_competition_l3774_377478

/-- Represents the penalty points for misses in a shooting competition -/
def penalty_points (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (n : ℚ) / 2 * (2 + (n - 1))

/-- The shooting competition problem -/
theorem shooting_competition_problem 
  (total_shots : ℕ) 
  (total_penalty : ℚ) 
  (hits : ℕ) : Prop :=
  total_shots = 25 ∧ 
  total_penalty = 7 ∧ 
  penalty_points (total_shots - hits) = total_penalty ∧
  hits = 21

/-- Proof of the shooting competition problem -/
theorem prove_shooting_competition : 
  ∃ (hits : ℕ), shooting_competition_problem 25 7 hits :=
sorry

end shooting_competition_problem_prove_shooting_competition_l3774_377478


namespace pluto_orbit_scientific_notation_l3774_377445

/-- The radius of Pluto's orbit in kilometers -/
def pluto_orbit_radius : ℝ := 5900000000

/-- The scientific notation representation of Pluto's orbit radius -/
def pluto_orbit_scientific : ℝ := 5.9 * (10 ^ 9)

/-- Theorem stating that the radius of Pluto's orbit is equal to its scientific notation representation -/
theorem pluto_orbit_scientific_notation : pluto_orbit_radius = pluto_orbit_scientific := by
  sorry

end pluto_orbit_scientific_notation_l3774_377445


namespace elliptical_machine_payment_l3774_377420

/-- Proves that the daily minimum payment for an elliptical machine is $6 given the specified conditions --/
theorem elliptical_machine_payment 
  (total_cost : ℝ) 
  (down_payment_ratio : ℝ) 
  (payment_days : ℕ) 
  (h1 : total_cost = 120) 
  (h2 : down_payment_ratio = 1/2) 
  (h3 : payment_days = 10) : 
  (total_cost * (1 - down_payment_ratio)) / payment_days = 6 := by
sorry

end elliptical_machine_payment_l3774_377420


namespace power_of_three_and_seven_hundreds_digit_l3774_377490

theorem power_of_three_and_seven_hundreds_digit : 
  ∃ (a b : ℕ), 
    100 ≤ 3^a ∧ 3^a < 1000 ∧
    100 ≤ 7^b ∧ 7^b < 1000 ∧
    (3^a / 100 % 10 = 7) ∧ (7^b / 100 % 10 = 7) := by
  sorry

end power_of_three_and_seven_hundreds_digit_l3774_377490


namespace starters_count_l3774_377495

/-- Represents a set of twins -/
structure TwinSet :=
  (twin1 : ℕ)
  (twin2 : ℕ)

/-- Represents a basketball team -/
structure BasketballTeam :=
  (total_players : ℕ)
  (twin_set1 : TwinSet)
  (twin_set2 : TwinSet)

/-- Calculates the number of ways to choose starters with twin restrictions -/
def choose_starters (team : BasketballTeam) (num_starters : ℕ) : ℕ :=
  sorry

/-- The specific basketball team in the problem -/
def problem_team : BasketballTeam :=
  { total_players := 18
  , twin_set1 := { twin1 := 1, twin2 := 2 }  -- Representing Ben & Jerry
  , twin_set2 := { twin1 := 3, twin2 := 4 }  -- Representing Tom & Tim
  }

theorem starters_count : choose_starters problem_team 5 = 1834 := by
  sorry

end starters_count_l3774_377495


namespace weight_sum_l3774_377467

/-- Given the weights of four people satisfying certain conditions, 
    prove that the sum of the first and fourth person's weights is 372 pounds. -/
theorem weight_sum (e f g h : ℝ) 
  (ef_sum : e + f = 320)
  (fg_sum : f + g = 298)
  (gh_sum : g + h = 350) :
  e + h = 372 := by
  sorry

end weight_sum_l3774_377467


namespace area_of_triangle_PQR_l3774_377435

/-- Given two lines intersecting at point P(2,5) with slopes 3 and -1 respectively,
    and forming a triangle PQR with the x-axis, prove that the area of triangle PQR is 25/3 -/
theorem area_of_triangle_PQR (P : ℝ × ℝ) (m₁ m₂ : ℝ) : 
  P = (2, 5) →
  m₁ = 3 →
  m₂ = -1 →
  let Q := (P.1 - P.2 / m₁, 0)
  let R := (P.1 + P.2 / m₂, 0)
  (1/2 : ℝ) * |R.1 - Q.1| * P.2 = 25/3 := by
  sorry

end area_of_triangle_PQR_l3774_377435


namespace jennys_money_l3774_377475

theorem jennys_money (original : ℚ) : 
  (original - (3/7 * original + 2/5 * original) = 24) → 
  (1/2 * original = 70) := by
sorry

end jennys_money_l3774_377475


namespace four_inch_cube_painted_faces_l3774_377470

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Represents a painted cube -/
structure PaintedCube extends Cube where
  paintedFaces : ℕ

/-- Calculates the number of smaller cubes with at least two painted faces
    when a painted cube is cut into unit cubes -/
def numCubesWithTwoPaintedFaces (c : PaintedCube) : ℕ :=
  sorry

theorem four_inch_cube_painted_faces :
  let bigCube : PaintedCube := ⟨⟨4⟩, 6⟩
  numCubesWithTwoPaintedFaces bigCube = 32 := by sorry

end four_inch_cube_painted_faces_l3774_377470


namespace arithmetic_sequence_sum_l3774_377472

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → (a 3 + a 4 + a 5 = 12) → (a 1 + a 7 = 8) :=
by
  sorry

end arithmetic_sequence_sum_l3774_377472


namespace tree_watering_l3774_377431

theorem tree_watering (num_boys : ℕ) (trees_per_boy : ℕ) :
  num_boys = 9 →
  trees_per_boy = 3 →
  num_boys * trees_per_boy = 27 :=
by sorry

end tree_watering_l3774_377431


namespace sufficient_not_necessary_l3774_377494

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ |x| ≥ 2) := by
  sorry

end sufficient_not_necessary_l3774_377494


namespace smallest_valid_seating_l3774_377415

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone already seated. -/
def valid_seating (table : CircularTable) : Prop :=
  table.seated_people > 0 ∧
  table.seated_people ≤ table.total_chairs ∧
  ∀ k : ℕ, k < table.total_chairs → 
    ∃ i j : ℕ, i < table.seated_people ∧ j < table.seated_people ∧
      (i * (table.total_chairs / table.seated_people) % table.total_chairs = k ∨
       j * (table.total_chairs / table.seated_people) % table.total_chairs = (k + 1) % table.total_chairs)

/-- The main theorem stating that 18 is the smallest number of people that can be validly seated. -/
theorem smallest_valid_seating :
  ∀ n : ℕ, n < 18 → ¬(valid_seating ⟨72, n⟩) ∧ 
  valid_seating ⟨72, 18⟩ := by
  sorry

#check smallest_valid_seating

end smallest_valid_seating_l3774_377415


namespace remainder_problem_l3774_377458

theorem remainder_problem (N : ℤ) (h : N % 296 = 75) : N % 37 = 1 := by
  sorry

end remainder_problem_l3774_377458


namespace cyclic_quadrilateral_angles_l3774_377497

-- Define a cyclic quadrilateral
def CyclicQuadrilateral (a b c d : ℝ) : Prop :=
  a + c = 180 ∧ b + d = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

-- Define an arithmetic progression
def ArithmeticProgression (a b c d : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ b - a = r ∧ c - b = r ∧ d - c = r

-- Define a geometric progression
def GeometricProgression (a b c d : ℝ) : Prop :=
  ∃ (q : ℝ), q ≠ 1 ∧ b / a = q ∧ c / b = q ∧ d / c = q

theorem cyclic_quadrilateral_angles :
  (∃ (a b c d : ℝ), CyclicQuadrilateral a b c d ∧ ArithmeticProgression a b c d) ∧
  (¬ ∃ (a b c d : ℝ), CyclicQuadrilateral a b c d ∧ GeometricProgression a b c d) :=
by sorry

end cyclic_quadrilateral_angles_l3774_377497


namespace sequence_length_bound_l3774_377448

theorem sequence_length_bound (N : ℕ) (m : ℕ) (a : ℕ → ℕ) :
  (∀ i j, 1 ≤ i → i < j → j ≤ m → a i < a j) →
  (∀ i, 1 ≤ i → i ≤ m → a i ≤ N) →
  (∀ i j, 1 ≤ i → i < j → j ≤ m → Nat.lcm (a i) (a j) ≤ N) →
  m ≤ 2 * Int.floor (Real.sqrt N) :=
by sorry

end sequence_length_bound_l3774_377448


namespace inscribed_squares_segment_product_l3774_377447

theorem inscribed_squares_segment_product :
  ∀ (small_area large_area : ℝ) (x : ℝ),
    small_area = 16 →
    large_area = 25 →
    x + 3*x = Real.sqrt large_area →
    x * (3*x) = 75/16 := by
  sorry

end inscribed_squares_segment_product_l3774_377447


namespace inequality_proof_l3774_377409

theorem inequality_proof (b : ℝ) (n : ℕ) (h1 : b > 0) (h2 : n > 2) :
  let floor_b := ⌊b⌋
  let d := ((floor_b + 1 - b) * floor_b) / (floor_b + 1)
  (d + n - 2) / (floor_b + n - 2) > (floor_b + n - 1 - b) / (floor_b + n - 1) := by
  sorry

end inequality_proof_l3774_377409


namespace point_on_x_axis_l3774_377424

/-- If a point M(a+3, a+1) lies on the x-axis, then its coordinates are (2,0) -/
theorem point_on_x_axis (a : ℝ) :
  (a + 1 = 0) →  -- Condition for M to be on x-axis
  ((a + 3, a + 1) : ℝ × ℝ) = (2, 0) := by
sorry

end point_on_x_axis_l3774_377424


namespace distance_to_destination_l3774_377437

/-- Proves that the distance to the destination is 2.25 kilometers given the specified conditions. -/
theorem distance_to_destination
  (rowing_speed : ℝ)
  (river_speed : ℝ)
  (round_trip_time : ℝ)
  (h1 : rowing_speed = 4)
  (h2 : river_speed = 2)
  (h3 : round_trip_time = 1.5)
  : ∃ (distance : ℝ),
    distance = 2.25 ∧
    round_trip_time = distance / (rowing_speed + river_speed) + distance / (rowing_speed - river_speed) :=
by
  sorry

end distance_to_destination_l3774_377437


namespace power_of_negative_square_l3774_377401

theorem power_of_negative_square (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end power_of_negative_square_l3774_377401


namespace min_sum_squares_l3774_377404

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (a b c d e f g h : Int)
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
  (he : e ∈ S) (hf : f ∈ S) (hg : g ∈ S) (hh : h ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
               f ≠ g ∧ f ≠ h ∧
               g ≠ h) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 8 :=
by sorry

end min_sum_squares_l3774_377404


namespace alloy_composition_l3774_377496

theorem alloy_composition (gold_weight copper_weight alloy_weight : ℝ) 
  (h1 : gold_weight = 19)
  (h2 : alloy_weight = 17)
  (h3 : (4 * gold_weight + copper_weight) / 5 = alloy_weight) : 
  copper_weight = 9 := by
  sorry

end alloy_composition_l3774_377496


namespace smaller_number_in_ratio_l3774_377487

theorem smaller_number_in_ratio (a b c d x y : ℝ) : 
  0 < a → a < b → 0 < d → d < c →
  x > 0 → y > 0 →
  x / y = a / b →
  x + y = c - d →
  d = 2 * x - y →
  min x y = (2 * a * c - b * c) / (3 * (2 * a - b)) := by
  sorry

end smaller_number_in_ratio_l3774_377487


namespace canteen_theorem_l3774_377408

/-- Represents the number of dishes available --/
def num_dishes : ℕ := 6

/-- Calculates the maximum number of days based on the number of dishes --/
def max_days (n : ℕ) : ℕ := 2^n

/-- Calculates the average number of dishes per day --/
def avg_dishes_per_day (n : ℕ) : ℚ := n / 2

theorem canteen_theorem :
  max_days num_dishes = 64 ∧ avg_dishes_per_day num_dishes = 3 := by sorry

end canteen_theorem_l3774_377408


namespace standard_deviation_double_data_l3774_377405

def data1 : List ℝ := [2, 3, 4, 5]
def data2 : List ℝ := [4, 6, 8, 10]

def standard_deviation (data : List ℝ) : ℝ := sorry

theorem standard_deviation_double_data :
  standard_deviation data1 = (1 / 2) * standard_deviation data2 := by sorry

end standard_deviation_double_data_l3774_377405


namespace incorrect_arrangements_count_l3774_377498

/-- The number of letters in the word --/
def word_length : ℕ := 4

/-- The total number of possible arrangements of the letters --/
def total_arrangements : ℕ := Nat.factorial word_length

/-- The number of correct arrangements (always 1 for a single word) --/
def correct_arrangements : ℕ := 1

/-- Theorem: The number of incorrect arrangements of a 4-letter word is 23 --/
theorem incorrect_arrangements_count :
  total_arrangements - correct_arrangements = 23 := by sorry

end incorrect_arrangements_count_l3774_377498


namespace empty_subset_of_A_l3774_377489

def A : Set ℤ := {x | 0 < x ∧ x < 3}

theorem empty_subset_of_A : ∅ ⊆ A := by
  sorry

end empty_subset_of_A_l3774_377489


namespace min_value_of_sum_l3774_377488

theorem min_value_of_sum (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) :
  let f := (x₁ + x₃) / (x₅ + 2*x₂ + 3*x₄) + (x₂ + x₄) / (x₁ + 2*x₃ + 3*x₅) + 
           (x₃ + x₅) / (x₂ + 2*x₄ + 3*x₁) + (x₄ + x₁) / (x₃ + 2*x₅ + 3*x₂) + 
           (x₅ + x₂) / (x₄ + 2*x₁ + 3*x₃)
  f ≥ 5/3 ∧ (f = 5/3 ↔ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) :=
by
  sorry

end min_value_of_sum_l3774_377488


namespace cylinder_generatrix_length_l3774_377469

/-- The length of the generatrix of a cylinder with base radius 1 and lateral surface area 6π is 2 -/
theorem cylinder_generatrix_length :
  ∀ (generatrix : ℝ),
  (generatrix > 0) →
  (2 * π * 1 + 2 * π * generatrix = 6 * π) →
  generatrix = 2 := by
sorry

end cylinder_generatrix_length_l3774_377469


namespace fourth_draw_probability_problem_solution_l3774_377473

/-- A box containing red and black balls -/
structure Box where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball from a box -/
def prob_black (b : Box) : ℚ :=
  b.black_balls / (b.red_balls + b.black_balls)

/-- The box described in the problem -/
def problem_box : Box :=
  { red_balls := 4, black_balls := 4 }

theorem fourth_draw_probability (b : Box) :
  prob_black b = 1 / 2 →
  (∀ n : ℕ, n > 0 → prob_black { red_balls := b.red_balls - min n b.red_balls,
                                 black_balls := b.black_balls - min n b.black_balls } = 1 / 2) →
  prob_black { red_balls := b.red_balls - min 3 b.red_balls,
               black_balls := b.black_balls - min 3 b.black_balls } = 1 / 2 :=
by sorry

theorem problem_solution :
  prob_black problem_box = 1 / 2 ∧
  (∀ n : ℕ, n > 0 → prob_black { red_balls := problem_box.red_balls - min n problem_box.red_balls,
                                 black_balls := problem_box.black_balls - min n problem_box.black_balls } = 1 / 2) :=
by sorry

end fourth_draw_probability_problem_solution_l3774_377473


namespace ball_count_l3774_377463

theorem ball_count (white green yellow red purple : ℕ)
  (h1 : white = 50)
  (h2 : green = 30)
  (h3 : yellow = 10)
  (h4 : red = 7)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 0.9) :
  white + green + yellow + red + purple = 100 := by
sorry

end ball_count_l3774_377463


namespace factorization_mx_plus_my_l3774_377403

theorem factorization_mx_plus_my (m x y : ℝ) : m * x + m * y = m * (x + y) := by
  sorry

end factorization_mx_plus_my_l3774_377403


namespace income_comparison_l3774_377492

/-- Given that Mart's income is 60% more than Tim's income, and Tim's income is 50% less than Juan's income, 
    prove that Mart's income is 80% of Juan's income. -/
theorem income_comparison (tim juan mart : ℝ) 
  (h1 : mart = tim + 0.6 * tim) 
  (h2 : tim = juan - 0.5 * juan) : 
  mart = 0.8 * juan := by sorry

end income_comparison_l3774_377492


namespace people_behind_yuna_l3774_377455

theorem people_behind_yuna (total : Nat) (in_front : Nat) (behind : Nat) : 
  total = 7 → in_front = 2 → behind = total - in_front - 1 → behind = 4 := by
  sorry

end people_behind_yuna_l3774_377455


namespace pascals_triangle_20th_row_5th_number_l3774_377461

theorem pascals_triangle_20th_row_5th_number : 
  let n : ℕ := 20  -- The row number (0-indexed)
  let k : ℕ := 4   -- The position in the row (0-indexed)
  Nat.choose n k = 4845 := by
sorry

end pascals_triangle_20th_row_5th_number_l3774_377461


namespace smallest_solution_cubic_equation_l3774_377465

theorem smallest_solution_cubic_equation :
  ∃ (x : ℝ), x = 2/3 ∧ 24 * x^3 - 106 * x^2 + 116 * x - 70 = 0 ∧
  ∀ (y : ℝ), 24 * y^3 - 106 * y^2 + 116 * y - 70 = 0 → y ≥ 2/3 :=
sorry

end smallest_solution_cubic_equation_l3774_377465


namespace no_charming_numbers_l3774_377474

def is_charming (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = a + b^3

theorem no_charming_numbers : ¬∃ (n : ℕ), is_charming n :=
sorry

end no_charming_numbers_l3774_377474


namespace custom_operation_equality_l3774_377443

/-- Custom operation ⊕ for real numbers -/
def circle_plus (a b : ℝ) : ℝ := (a + b) ^ 2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y : ℝ) : 
  circle_plus (circle_plus ((x + y) ^ 2) ((y + x) ^ 2)) 2 = 4 * ((x + y) ^ 2 + 1) ^ 2 := by
  sorry

end custom_operation_equality_l3774_377443


namespace distance_sum_squares_constant_l3774_377414

/-- Two concentric circles with radii R₁ and R₂ -/
structure ConcentricCircles (R₁ R₂ : ℝ) where
  center : ℝ × ℝ

/-- A point on a circle -/
structure PointOnCircle (c : ℝ × ℝ) (R : ℝ) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.1)^2 + (point.2 - c.2)^2 = R^2

/-- A diameter of a circle -/
structure Diameter (c : ℝ × ℝ) (R : ℝ) where
  endpointA : ℝ × ℝ
  endpointB : ℝ × ℝ
  is_diameter : (endpointA.1 - c.1)^2 + (endpointA.2 - c.2)^2 = R^2 ∧
                (endpointB.1 - c.1)^2 + (endpointB.2 - c.2)^2 = R^2 ∧
                (endpointA.1 - endpointB.1)^2 + (endpointA.2 - endpointB.2)^2 = 4 * R^2

/-- The theorem statement -/
theorem distance_sum_squares_constant
  (R₁ R₂ : ℝ) (circles : ConcentricCircles R₁ R₂)
  (C : PointOnCircle circles.center R₂)
  (AB : Diameter circles.center R₁) :
  let distAC := ((AB.endpointA.1 - C.point.1)^2 + (AB.endpointA.2 - C.point.2)^2)
  let distBC := ((AB.endpointB.1 - C.point.1)^2 + (AB.endpointB.2 - C.point.2)^2)
  distAC + distBC = 2 * R₁^2 + 2 * R₂^2 := by
  sorry

end distance_sum_squares_constant_l3774_377414


namespace coefficient_a3b3_value_l3774_377459

/-- The coefficient of a^3b^3 in (a+b)^6(c+1/c)^8 -/
def coefficient_a3b3 (a b c : ℝ) : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 8 4)

theorem coefficient_a3b3_value :
  ∀ a b c : ℝ, coefficient_a3b3 a b c = 1400 := by
  sorry

end coefficient_a3b3_value_l3774_377459


namespace complement_union_theorem_l3774_377422

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A
def A : Finset Nat := {1, 2}

-- Define set B
def B : Finset Nat := {2, 3}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3} := by sorry

end complement_union_theorem_l3774_377422


namespace misha_max_cities_l3774_377417

/-- The maximum number of cities Misha can visit -/
def max_cities_visited (n k : ℕ) : ℕ :=
  if k ≥ n - 3 then min (n - k) 2 else n - k

/-- Theorem stating the maximum number of cities Misha can visit -/
theorem misha_max_cities (n k : ℕ) (h1 : n ≥ 2) (h2 : k ≥ 1) :
  max_cities_visited n k = 
    if k ≥ n - 3 then min (n - k) 2 else n - k :=
by sorry

end misha_max_cities_l3774_377417


namespace part_one_part_two_l3774_377407

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x > 5}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 4}

-- Part 1
theorem part_one :
  (A ∩ B 2 = {x | 5 < x ∧ x < 6}) ∧
  (Set.univ \ A = {x | -1 ≤ x ∧ x ≤ 5}) :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  B a ⊆ (Set.univ \ A) ↔ a ∈ Set.Iic 3 ∪ Set.Ici 5 :=
sorry

end part_one_part_two_l3774_377407


namespace double_mean_value_function_range_l3774_377453

/-- Definition of a double mean value function -/
def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (deriv^[2] f x₁ = (f b - f a) / (b - a)) ∧
    (deriv^[2] f x₂ = (f b - f a) / (b - a))

/-- The main theorem -/
theorem double_mean_value_function_range (a : ℝ) (m : ℝ) :
  is_double_mean_value_function (fun x => 2 * x^3 - x^2 + m) 0 (2 * a) →
  1/8 < a ∧ a < 1/4 := by
  sorry

end double_mean_value_function_range_l3774_377453


namespace valid_grid_count_l3774_377499

/-- A type representing a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Checks if a grid is valid according to the problem rules -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, j < 2 → g i j < g i (j+1)) ∧  -- rows in ascending order
  (∀ i j, i < 2 → g i j < g (i+1) j) ∧  -- columns in ascending order
  (∀ i j, g i j ∈ Finset.range 9) ∧     -- numbers from 1 to 9
  (g 0 0 = 1) ∧ (g 1 1 = 4) ∧ (g 2 2 = 9)  -- pre-filled numbers

/-- The set of all valid grids -/
def valid_grids : Finset Grid :=
  sorry

theorem valid_grid_count : Finset.card valid_grids = 12 := by
  sorry

end valid_grid_count_l3774_377499


namespace sum_equality_l3774_377450

theorem sum_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d)
  (sum_nonzero : a + b ≠ 0)
  (product_eq : a * c = b * d) : 
  a + c = b + d := by
  sorry

end sum_equality_l3774_377450


namespace circle_radius_increment_l3774_377460

theorem circle_radius_increment (c₁ c₂ : ℝ) (h₁ : c₁ = 50) (h₂ : c₂ = 60) :
  c₂ / (2 * Real.pi) - c₁ / (2 * Real.pi) = 5 / Real.pi := by
  sorry

end circle_radius_increment_l3774_377460


namespace pencils_left_over_l3774_377484

theorem pencils_left_over (total_pencils : ℕ) (students_class1 : ℕ) (students_class2 : ℕ) 
  (h1 : total_pencils = 210)
  (h2 : students_class1 = 30)
  (h3 : students_class2 = 20) :
  total_pencils - (students_class1 + students_class2) * (total_pencils / (students_class1 + students_class2)) = 10 := by
  sorry

end pencils_left_over_l3774_377484


namespace largest_n_with_triangle_property_l3774_377446

/-- A set of consecutive positive integers has the triangle property for all 9-element subsets -/
def has_triangle_property (s : Set ℕ) : Prop :=
  ∀ (x y z : ℕ), x ∈ s → y ∈ s → z ∈ s → x < y → y < z → z < x + y

/-- The set of consecutive positive integers from 6 to n -/
def consecutive_set (n : ℕ) : Set ℕ :=
  {x : ℕ | 6 ≤ x ∧ x ≤ n}

/-- The theorem stating that 224 is the largest possible value of n -/
theorem largest_n_with_triangle_property :
  ∀ n : ℕ, (has_triangle_property (consecutive_set n)) → n ≤ 224 :=
sorry

end largest_n_with_triangle_property_l3774_377446


namespace sqrt_of_negative_nine_squared_l3774_377426

theorem sqrt_of_negative_nine_squared : Real.sqrt ((-9)^2) = 9 := by
  sorry

end sqrt_of_negative_nine_squared_l3774_377426


namespace unique_solution_condition_l3774_377441

/-- A system of equations has exactly one solution if and only if a = 2 and b = -1 -/
theorem unique_solution_condition (a b : ℝ) : 
  (∃! x y, y = x^2 ∧ y = 2*x + b) ↔ (a = 2 ∧ b = -1) :=
sorry

end unique_solution_condition_l3774_377441


namespace platform_length_l3774_377423

/-- Given a train of length 600 meters that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, prove that the platform length is 700 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ) :
  train_length = 600 ∧ 
  platform_cross_time = 39 ∧ 
  pole_cross_time = 18 →
  (train_length + (train_length / pole_cross_time * platform_cross_time - train_length)) = 700 :=
by sorry

end platform_length_l3774_377423


namespace class_size_ratio_l3774_377449

/-- Given three classes A, B, and C, prove that the ratio of the size of Class A to Class C is 1/3 -/
theorem class_size_ratio (size_A size_B size_C : ℕ) : 
  size_A = 2 * size_B → 
  size_B = 20 → 
  size_C = 120 → 
  (size_A : ℚ) / size_C = 1 / 3 := by
  sorry

end class_size_ratio_l3774_377449


namespace congruence_solutions_count_l3774_377430

theorem congruence_solutions_count :
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, x > 0 ∧ x < 120 ∧ (x + 17) % 38 = 75 % 38) ∧
    (∀ x : ℕ, x > 0 ∧ x < 120 ∧ (x + 17) % 38 = 75 % 38 → x ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end congruence_solutions_count_l3774_377430


namespace traditionalist_fraction_l3774_377440

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) 
  (total_progressives : ℚ) :
  num_provinces = 6 →
  num_traditionalists_per_province = total_progressives / 9 →
  (num_provinces : ℚ) * num_traditionalists_per_province / 
    (total_progressives + (num_provinces : ℚ) * num_traditionalists_per_province) = 2 / 5 :=
by sorry

end traditionalist_fraction_l3774_377440


namespace sequence_is_increasing_l3774_377438

def isIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem sequence_is_increasing (a : ℕ → ℝ) 
    (h : ∀ n, a (n + 1) - a n - 3 = 0) : 
    isIncreasing a := by
  sorry

end sequence_is_increasing_l3774_377438


namespace smaller_number_in_ratio_l3774_377421

theorem smaller_number_in_ratio (a b : ℝ) : 
  a / b = 3 / 4 → a + b = 420 → a = 180 := by
  sorry

end smaller_number_in_ratio_l3774_377421


namespace total_pencils_l3774_377454

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) 
  (h1 : jessica_pencils = 8)
  (h2 : sandy_pencils = 8)
  (h3 : jason_pencils = 8) :
  jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end total_pencils_l3774_377454


namespace min_value_of_m_minus_n_l3774_377413

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1

noncomputable def g (x : ℝ) : ℝ := 2 * Real.exp (x - 1/2)

theorem min_value_of_m_minus_n (m n : ℝ) (h : f m = g n) :
  ∃ (k : ℝ), k = 1/2 + Real.log 2 ∧ m - n ≥ k := by
  sorry

end min_value_of_m_minus_n_l3774_377413


namespace correct_units_l3774_377452

/-- Represents the number of units in a building project -/
structure BuildingProject where
  first_building : ℕ
  second_building : ℕ
  third_building : ℕ
  apartments : ℕ
  condos : ℕ
  townhouses : ℕ
  bungalows : ℕ

/-- Calculates the correct number of units for each type in the building project -/
def calculate_units (project : BuildingProject) : Prop :=
  -- First building conditions
  project.first_building = 4000 ∧
  project.apartments ≥ 2000 ∧
  project.condos ≥ 2000 ∧
  -- Second building conditions
  project.second_building = (2 : ℕ) * project.first_building / 5 ∧
  -- Third building conditions
  project.third_building = (6 : ℕ) * project.second_building / 5 ∧
  project.townhouses = (3 : ℕ) * project.third_building / 5 ∧
  project.bungalows = (2 : ℕ) * project.third_building / 5 ∧
  -- Total units calculation
  project.apartments = 3200 ∧
  project.condos = 2400 ∧
  project.townhouses = 1152 ∧
  project.bungalows = 768

/-- Theorem stating that the calculated units are correct -/
theorem correct_units (project : BuildingProject) : 
  calculate_units project → 
  project.apartments = 3200 ∧ 
  project.condos = 2400 ∧ 
  project.townhouses = 1152 ∧ 
  project.bungalows = 768 := by
  sorry

end correct_units_l3774_377452


namespace complex_equation_proof_l3774_377428

/-- Given the complex equation (2+i)/(i+1) - 2i = a + bi, prove that b - ai = -5/2 - 3/2i --/
theorem complex_equation_proof (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 + i) / (i + 1) - 2 * i = a + b * i) : 
  b - a * i = -5/2 - 3/2 * i :=
by sorry

end complex_equation_proof_l3774_377428


namespace income_minus_expenses_tax_lower_l3774_377442

/-- Represents the tax options available --/
inductive TaxOption
  | IncomeTax
  | IncomeMinusExpensesTax

/-- Calculates the tax payable for a given option --/
def calculateTax (option : TaxOption) (totalIncome expenses insuranceContributions : ℕ) : ℕ :=
  match option with
  | TaxOption.IncomeTax =>
      let incomeTax := totalIncome * 6 / 100
      let maxDeduction := min (incomeTax / 2) insuranceContributions
      incomeTax - maxDeduction
  | TaxOption.IncomeMinusExpensesTax =>
      let taxBase := totalIncome - expenses
      let regularTax := taxBase * 15 / 100
      let minimumTax := totalIncome * 1 / 100
      max regularTax minimumTax

/-- Theorem stating that the Income minus expenses tax option results in lower tax --/
theorem income_minus_expenses_tax_lower
  (totalIncome expenses insuranceContributions : ℕ)
  (h1 : totalIncome = 150000000)
  (h2 : expenses = 141480000)
  (h3 : insuranceContributions = 16560000) :
  calculateTax TaxOption.IncomeMinusExpensesTax totalIncome expenses insuranceContributions <
  calculateTax TaxOption.IncomeTax totalIncome expenses insuranceContributions :=
by
  sorry


end income_minus_expenses_tax_lower_l3774_377442


namespace clever_calculation_l3774_377468

theorem clever_calculation :
  (46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 = 56.056) ∧
  (101 * 92 - 92 = 9200) ∧
  (36000 / 125 / 8 = 36) := by
  sorry

end clever_calculation_l3774_377468


namespace port_perry_wellington_ratio_l3774_377406

/-- The ratio of Port Perry's population to Wellington's population -/
def population_ratio (port_perry : ℕ) (wellington : ℕ) (lazy_harbor : ℕ) : ℚ :=
  port_perry / wellington

theorem port_perry_wellington_ratio :
  ∀ (port_perry wellington lazy_harbor : ℕ),
    wellington = 900 →
    port_perry = lazy_harbor + 800 →
    port_perry + lazy_harbor = 11800 →
    population_ratio port_perry wellington lazy_harbor = 7 := by
  sorry

#check port_perry_wellington_ratio

end port_perry_wellington_ratio_l3774_377406


namespace length_of_AB_l3774_377466

/-- Given two points P and Q on a line segment AB, prove that AB has length 35 -/
theorem length_of_AB (A B P Q : ℝ) : 
  (0 < A ∧ A < P ∧ P < Q ∧ Q < B) →  -- P and Q are on AB and on the same side of midpoint
  (P - A) / (B - P) = 1 / 4 →        -- P divides AB in ratio 1:4
  (Q - A) / (B - Q) = 2 / 5 →        -- Q divides AB in ratio 2:5
  Q - P = 3 →                        -- Distance PQ = 3
  B - A = 35 := by                   -- Length of AB is 35
sorry


end length_of_AB_l3774_377466


namespace ones_12_div_13_ones_16_div_17_l3774_377432

/-- The number formed by n consecutive ones -/
def ones (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Theorem: The number formed by 12 consecutive ones is divisible by 13 -/
theorem ones_12_div_13 : 13 ∣ ones 12 := by sorry

/-- Theorem: The number formed by 16 consecutive ones is divisible by 17 -/
theorem ones_16_div_17 : 17 ∣ ones 16 := by sorry

end ones_12_div_13_ones_16_div_17_l3774_377432


namespace pizza_equivalents_theorem_l3774_377400

/-- Calculates the total quantity of pizza equivalents served -/
def total_pizza_equivalents (lunch_pizzas : ℕ) (dinner_pizzas : ℕ) (lunch_calzones : ℕ) : ℕ :=
  lunch_pizzas + dinner_pizzas + (lunch_calzones / 2)

/-- Proves that the total quantity of pizza equivalents served is 17 -/
theorem pizza_equivalents_theorem :
  total_pizza_equivalents 9 6 4 = 17 := by
  sorry

end pizza_equivalents_theorem_l3774_377400


namespace magazine_cost_l3774_377411

theorem magazine_cost (total_books : ℕ) (num_magazines : ℕ) (book_cost : ℕ) (total_spent : ℕ) : 
  total_books = 16 → 
  num_magazines = 3 → 
  book_cost = 11 → 
  total_spent = 179 → 
  (total_spent - total_books * book_cost) / num_magazines = 1 := by
sorry

end magazine_cost_l3774_377411


namespace square_property_implies_zero_l3774_377434

theorem square_property_implies_zero (a b : ℤ) : 
  (∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) → a = 0 :=
by sorry

end square_property_implies_zero_l3774_377434


namespace jack_and_jill_speed_l3774_377481

theorem jack_and_jill_speed : 
  ∀ x : ℝ, 
    x ≠ -2 →
    (x^2 - 7*x - 12 = (x^2 - 3*x - 10) / (x + 2)) →
    (x^2 - 7*x - 12 = 2) := by
  sorry

end jack_and_jill_speed_l3774_377481


namespace cone_volume_from_circle_sector_l3774_377416

/-- The volume of a cone formed by rolling up a three-quarter sector of a circle -/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 4) :
  let sector_angle : ℝ := 3 * π / 2
  let base_radius : ℝ := sector_angle * r / (2 * π)
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * cone_height = 3 * π * Real.sqrt 7 := by
  sorry

end cone_volume_from_circle_sector_l3774_377416


namespace red_spot_percentage_is_40_l3774_377425

/-- Represents the farm with cows and their spot characteristics -/
structure Farm where
  total_cows : ℕ
  no_spot_cows : ℕ
  blue_spot_ratio : ℚ

/-- Calculates the percentage of cows with a red spot -/
def red_spot_percentage (farm : Farm) : ℚ :=
  let no_red_spot := farm.no_spot_cows / farm.blue_spot_ratio
  let red_spot := farm.total_cows - no_red_spot
  (red_spot / farm.total_cows) * 100

/-- Theorem stating that for the given farm conditions, 
    the percentage of cows with a red spot is 40% -/
theorem red_spot_percentage_is_40 (farm : Farm) 
  (h1 : farm.total_cows = 140)
  (h2 : farm.no_spot_cows = 63)
  (h3 : farm.blue_spot_ratio = 3/4) :
  red_spot_percentage farm = 40 := by
  sorry

#eval red_spot_percentage ⟨140, 63, 3/4⟩

end red_spot_percentage_is_40_l3774_377425


namespace quadratic_symmetry_l3774_377462

theorem quadratic_symmetry (a : ℝ) :
  (∃ (a : ℝ), 4 = a * (-2)^2) → (∃ (a : ℝ), 4 = a * 2^2) :=
by sorry

end quadratic_symmetry_l3774_377462


namespace power_twenty_equals_R_S_l3774_377436

theorem power_twenty_equals_R_S (a b : ℤ) (R S : ℝ) 
  (hR : R = (4 : ℝ) ^ a) 
  (hS : S = (5 : ℝ) ^ b) : 
  (20 : ℝ) ^ (a * b) = R ^ b * S ^ a := by sorry

end power_twenty_equals_R_S_l3774_377436


namespace rectangle_area_l3774_377480

theorem rectangle_area (w : ℝ) (l : ℝ) (A : ℝ) (P : ℝ) : 
  l = w + 6 →
  A = w * l →
  P = 2 * (w + l) →
  A = 2 * P →
  w = 3 →
  A = 27 :=
by sorry

end rectangle_area_l3774_377480


namespace probability_of_drawing_red_ball_l3774_377433

theorem probability_of_drawing_red_ball (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) :
  total_balls = red_balls + black_balls →
  red_balls = 3 →
  black_balls = 3 →
  (red_balls : ℚ) / (total_balls : ℚ) = 1/2 :=
by sorry

end probability_of_drawing_red_ball_l3774_377433


namespace total_books_l3774_377486

theorem total_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 6)
  (h2 : mystery_shelves = 5)
  (h3 : picture_shelves = 4) :
  books_per_shelf * mystery_shelves + books_per_shelf * picture_shelves = 54 := by
  sorry

end total_books_l3774_377486


namespace equilateral_roots_ratio_l3774_377444

/-- Given complex numbers z₁ and z₂ that are roots of z² + pz + q = 0,
    where p and q are complex numbers, and 0, z₁, and z₂ form an
    equilateral triangle in the complex plane, then p²/q = 1. -/
theorem equilateral_roots_ratio (p q z₁ z₂ : ℂ) :
  z₁^2 + p*z₁ + q = 0 →
  z₂^2 + p*z₂ + q = 0 →
  ∃ (ω : ℂ), ω^3 = 1 ∧ ω ≠ 1 ∧ z₂ = ω * z₁ →
  p^2 / q = 1 := by
  sorry

end equilateral_roots_ratio_l3774_377444


namespace nail_count_l3774_377412

/-- Given that Violet has 3 more than twice as many nails as Tickletoe and Violet has 27 nails, 
    prove that the total number of nails they have together is 39. -/
theorem nail_count (tickletoe_nails : ℕ) : 
  (2 * tickletoe_nails + 3 = 27) → (tickletoe_nails + 27 = 39) := by
  sorry

end nail_count_l3774_377412


namespace initial_oranges_count_l3774_377477

/-- The number of oranges Susan took from the box -/
def oranges_taken : ℕ := 35

/-- The number of oranges left in the box -/
def oranges_left : ℕ := 20

/-- The initial number of oranges in the box -/
def initial_oranges : ℕ := oranges_taken + oranges_left

theorem initial_oranges_count : initial_oranges = 55 := by
  sorry

end initial_oranges_count_l3774_377477
