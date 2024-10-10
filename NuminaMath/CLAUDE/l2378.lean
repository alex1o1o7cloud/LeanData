import Mathlib

namespace boat_speed_in_still_water_l2378_237899

/-- Proves that a boat's speed in still water is 51 kmph given the conditions -/
theorem boat_speed_in_still_water 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (stream_speed : ℝ) 
  (h1 : upstream_time = 2 * downstream_time)
  (h2 : stream_speed = 17) : 
  ∃ (boat_speed : ℝ), boat_speed = 51 ∧ 
    (boat_speed + stream_speed) * downstream_time = 
    (boat_speed - stream_speed) * upstream_time := by
  sorry

#check boat_speed_in_still_water

end boat_speed_in_still_water_l2378_237899


namespace hakimi_age_l2378_237886

/-- Given three friends with an average age of 40, where Jared is ten years older than Hakimi
    and Molly is 30 years old, prove that Hakimi's age is 40. -/
theorem hakimi_age (average_age : ℝ) (molly_age : ℝ) (jared_hakimi_age_diff : ℝ) 
  (h1 : average_age = 40)
  (h2 : molly_age = 30)
  (h3 : jared_hakimi_age_diff = 10) : 
  ∃ (hakimi_age : ℝ), hakimi_age = 40 ∧ 
    (hakimi_age + (hakimi_age + jared_hakimi_age_diff) + molly_age) / 3 = average_age :=
by sorry

end hakimi_age_l2378_237886


namespace bathtub_guests_l2378_237892

/-- Proves that given a bathtub with 10 liters capacity, after 3 guests use 1.5 liters each
    and 1 guest uses 1.75 liters, the remaining water can be used by exactly 3 more guests
    if each uses 1.25 liters. -/
theorem bathtub_guests (bathtub_capacity : ℝ) (guests_1 : ℕ) (water_1 : ℝ)
                        (guests_2 : ℕ) (water_2 : ℝ) (water_per_remaining_guest : ℝ) :
  bathtub_capacity = 10 →
  guests_1 = 3 →
  water_1 = 1.5 →
  guests_2 = 1 →
  water_2 = 1.75 →
  water_per_remaining_guest = 1.25 →
  (bathtub_capacity - (guests_1 * water_1 + guests_2 * water_2)) / water_per_remaining_guest = 3 :=
by sorry

end bathtub_guests_l2378_237892


namespace film_product_unique_l2378_237871

/-- Represents the alphabet-to-number mapping -/
def letter_value (c : Char) : Nat :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14
  | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _ => 0

/-- Calculates the product of letter values for a given string -/
def string_product (s : String) : Nat :=
  s.data.foldl (fun acc c => acc * letter_value c) 1

/-- Checks if a string is a valid four-letter combination (all uppercase letters) -/
def is_valid_combination (s : String) : Bool :=
  s.length = 4 && s.data.all (fun c => 'A' ≤ c && c ≤ 'Z')

/-- Theorem: The product of "FILM" is unique among all four-letter combinations -/
theorem film_product_unique :
  ∀ s : String, is_valid_combination s → s ≠ "FILM" →
  string_product s ≠ string_product "FILM" :=
sorry


end film_product_unique_l2378_237871


namespace cow_count_is_sixteen_l2378_237849

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem: If the total number of legs is 32 more than twice the number of heads,
    then the number of cows is 16 -/
theorem cow_count_is_sixteen (count : AnimalCount) :
  totalLegs count = 2 * totalHeads count + 32 → count.cows = 16 := by
  sorry


end cow_count_is_sixteen_l2378_237849


namespace union_of_M_and_N_l2378_237884

open Set

def M : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x : ℝ | x * (x - 1) ≤ 0}

theorem union_of_M_and_N : M ∪ N = Ioo (-1/2 : ℝ) 1 := by sorry

end union_of_M_and_N_l2378_237884


namespace short_trees_to_plant_l2378_237877

theorem short_trees_to_plant (current_short_trees : ℕ) (total_short_trees_after : ℕ) 
  (h1 : current_short_trees = 112)
  (h2 : total_short_trees_after = 217) :
  total_short_trees_after - current_short_trees = 105 := by
  sorry

#check short_trees_to_plant

end short_trees_to_plant_l2378_237877


namespace joseph_cards_l2378_237840

/-- Calculates the total number of cards Joseph had initially -/
def total_cards (num_students : ℕ) (cards_per_student : ℕ) (cards_left : ℕ) : ℕ :=
  num_students * cards_per_student + cards_left

/-- Proves that Joseph had 357 cards initially -/
theorem joseph_cards : total_cards 15 23 12 = 357 := by
  sorry

end joseph_cards_l2378_237840


namespace box_triples_count_l2378_237854

/-- The number of ordered triples (a, b, c) of positive integers satisfying the box conditions -/
def box_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 4 * (a * b + a * c + b * c))
    (Finset.product (Finset.range 100) (Finset.product (Finset.range 100) (Finset.range 100)))).card

/-- Theorem stating that there are exactly 2 ordered triples satisfying the box conditions -/
theorem box_triples_count : box_triples = 2 := by
  sorry

end box_triples_count_l2378_237854


namespace emerald_woods_circuit_length_l2378_237880

/-- Proves that the total length of the Emerald Woods Circuit is 43 miles given the hiking conditions --/
theorem emerald_woods_circuit_length :
  ∀ (a b c d e : ℝ),
    a + b + c = 28 →
    c + d = 24 →
    c + d + e = 39 →
    a + d = 30 →
    a + b + c + d + e = 43 := by
  sorry

end emerald_woods_circuit_length_l2378_237880


namespace smaller_paintings_count_l2378_237846

/-- Represents a museum with paintings and artifacts --/
structure Museum where
  total_wings : ℕ
  painting_wings : ℕ
  artifacts_per_wing : ℕ
  artifact_painting_ratio : ℕ

/-- The number of smaller paintings in each of the two wings --/
def smaller_paintings_per_wing (m : Museum) : ℕ :=
  ((m.artifacts_per_wing * (m.total_wings - m.painting_wings)) / m.artifact_painting_ratio - 1) / 2

/-- Theorem stating the number of smaller paintings per wing --/
theorem smaller_paintings_count (m : Museum) 
  (h1 : m.total_wings = 8)
  (h2 : m.painting_wings = 3)
  (h3 : m.artifacts_per_wing = 20)
  (h4 : m.artifact_painting_ratio = 4) :
  smaller_paintings_per_wing m = 12 := by
  sorry

end smaller_paintings_count_l2378_237846


namespace circle_equations_l2378_237883

-- Define the parallel lines
def line1 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + y + Real.sqrt 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 2 * Real.sqrt 2 * a = 0

-- Define the circle N
def circleN (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define point B
def pointB : ℝ × ℝ := (3, -2)

-- Define the line of symmetry
def lineSymmetry (x : ℝ) : Prop := x = -1

-- Define point C
def pointC : ℝ × ℝ := (-5, -2)

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 5)^2 + (y + 2)^2 = 49

-- Theorem statement
theorem circle_equations :
  ∃ (a : ℝ),
    (∀ x y, line1 a x y ↔ line2 a x y) →
    (∀ x y, circleN x y) →
    (pointC.1 = -pointB.1 - 2 ∧ pointC.2 = pointB.2) →
    (∀ x y, circleC x y) ∧
    (∃ x y, circleN x y ∧ circleC x y ∧
      (x - 3)^2 + (y - 4)^2 + ((x + 5)^2 + (y + 2)^2).sqrt = 10) :=
by sorry

end circle_equations_l2378_237883


namespace charlie_dana_difference_l2378_237808

/-- Represents the number of games won by each player -/
structure GameWins where
  perry : ℕ
  dana : ℕ
  charlie : ℕ
  phil : ℕ

/-- The conditions of the golf game results -/
def golf_results (g : GameWins) : Prop :=
  g.perry = g.dana + 5 ∧
  g.charlie < g.dana ∧
  g.phil = g.charlie + 3 ∧
  g.phil = 12 ∧
  g.perry = g.phil + 4

theorem charlie_dana_difference (g : GameWins) (h : golf_results g) :
  g.dana - g.charlie = 2 := by
  sorry

end charlie_dana_difference_l2378_237808


namespace factorization_2x_squared_minus_4x_l2378_237817

theorem factorization_2x_squared_minus_4x (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end factorization_2x_squared_minus_4x_l2378_237817


namespace equation_solution_l2378_237821

theorem equation_solution : ∃ x : ℕ, 16^5 + 16^5 + 16^5 = 4^x ∧ x = 20 := by
  sorry

end equation_solution_l2378_237821


namespace max_trip_weight_is_750_l2378_237867

/-- Represents the number of crates on a trip -/
inductive NumCrates
  | three
  | four
  | five

/-- The minimum weight of a single crate in kg -/
def minCrateWeight : ℝ := 150

/-- Calculates the maximum weight of crates on a single trip -/
def maxTripWeight (n : NumCrates) : ℝ :=
  match n with
  | .three => 3 * minCrateWeight
  | .four => 4 * minCrateWeight
  | .five => 5 * minCrateWeight

/-- Theorem: The maximum weight of crates on a single trip is 750 kg -/
theorem max_trip_weight_is_750 :
  ∀ n : NumCrates, maxTripWeight n ≤ 750 ∧ ∃ m : NumCrates, maxTripWeight m = 750 :=
by sorry

end max_trip_weight_is_750_l2378_237867


namespace expression_simplification_l2378_237815

theorem expression_simplification (m : ℝ) (h : m = 5) :
  (3*m + 6) / (m^2 + 4*m + 4) / ((m - 2) / (m + 2)) + 1 / (2 - m) = 2/3 := by
  sorry

end expression_simplification_l2378_237815


namespace geometric_sequence_product_l2378_237800

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 9 = 16 →
  a 2 * a 5 * a 8 = 64 := by
  sorry

#check geometric_sequence_product

end geometric_sequence_product_l2378_237800


namespace square_area_from_diagonal_l2378_237824

theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) : 
  diagonal = 12 → area = diagonal^2 / 2 → area = 72 := by
  sorry

#check square_area_from_diagonal

end square_area_from_diagonal_l2378_237824


namespace okeydokey_investment_l2378_237888

/-- Represents the investment scenario for earthworms -/
structure EarthwormInvestment where
  total_earthworms : ℕ
  artichokey_apples : ℕ
  okeydokey_earthworms : ℕ

/-- Calculates the number of apples Okeydokey invested -/
def okeydokey_apples (investment : EarthwormInvestment) : ℕ :=
  (investment.okeydokey_earthworms * (investment.artichokey_apples + investment.okeydokey_earthworms)) / 
  (investment.total_earthworms - investment.okeydokey_earthworms)

/-- Theorem stating that Okeydokey invested 5 apples -/
theorem okeydokey_investment (investment : EarthwormInvestment) 
  (h1 : investment.total_earthworms = 60)
  (h2 : investment.artichokey_apples = 7)
  (h3 : investment.okeydokey_earthworms = 25) : 
  okeydokey_apples investment = 5 := by
  sorry

#eval okeydokey_apples { total_earthworms := 60, artichokey_apples := 7, okeydokey_earthworms := 25 }

end okeydokey_investment_l2378_237888


namespace parallel_lines_imply_a_equals_3_l2378_237813

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

theorem parallel_lines_imply_a_equals_3 :
  ∀ a : ℝ,
  let l1 : Line := ⟨a, 2, 3*a⟩
  let l2 : Line := ⟨3, a-1, a-7⟩
  parallel l1 l2 → a = 3 := by
  sorry

#check parallel_lines_imply_a_equals_3

end parallel_lines_imply_a_equals_3_l2378_237813


namespace speedster_convertible_fraction_l2378_237843

theorem speedster_convertible_fraction :
  ∀ (total_inventory : ℕ) (speedsters : ℕ) (speedster_convertibles : ℕ),
    speedsters = total_inventory / 3 →
    total_inventory - speedsters = 30 →
    speedster_convertibles = 12 →
    (speedster_convertibles : ℚ) / speedsters = 4 / 5 := by
  sorry

end speedster_convertible_fraction_l2378_237843


namespace smallest_number_with_55_divisors_l2378_237805

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_55_divisors :
  ∃ (n : ℕ), num_divisors n = 55 ∧ 
  (∀ m : ℕ, num_divisors m = 55 → n ≤ m) ∧
  n = 3^4 * 2^10 := by
  sorry

end smallest_number_with_55_divisors_l2378_237805


namespace class_division_l2378_237826

theorem class_division (total_students : ℕ) (x : ℕ) : 
  (total_students = 8 * x + 2) ∧ (total_students = 9 * x - 4) → x = 6 := by
sorry

end class_division_l2378_237826


namespace min_value_sum_squares_l2378_237810

theorem min_value_sum_squares (x y z : ℝ) (h : x - 2*y - 3*z = 4) :
  ∃ (m : ℝ), m = 8/7 ∧ (∀ x y z : ℝ, x - 2*y - 3*z = 4 → x^2 + y^2 + z^2 ≥ m) ∧
  (∃ x y z : ℝ, x - 2*y - 3*z = 4 ∧ x^2 + y^2 + z^2 = m) := by
  sorry

end min_value_sum_squares_l2378_237810


namespace scott_distance_l2378_237862

/-- Given a 100-meter race where Scott runs 4 meters for every 5 meters that Chris runs,
    prove that Scott will have run 80 meters when Chris crosses the finish line. -/
theorem scott_distance (race_length : ℕ) (scott_ratio chris_ratio : ℕ) : 
  race_length = 100 →
  scott_ratio = 4 →
  chris_ratio = 5 →
  (scott_ratio * race_length) / chris_ratio = 80 := by
sorry

end scott_distance_l2378_237862


namespace barrys_age_l2378_237831

theorem barrys_age (sisters_average_age : ℕ) (total_average_age : ℕ) : 
  sisters_average_age = 27 → total_average_age = 28 → 
  (3 * sisters_average_age + 31) / 4 = total_average_age :=
by
  sorry

#check barrys_age

end barrys_age_l2378_237831


namespace f_is_direct_proportion_l2378_237866

/-- Definition of direct proportion --/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function we want to prove is directly proportional --/
def f (x : ℝ) : ℝ := -0.1 * x

/-- Theorem stating that f is a direct proportion --/
theorem f_is_direct_proportion : is_direct_proportion f := by
  sorry

end f_is_direct_proportion_l2378_237866


namespace sequence_2015th_term_l2378_237825

/-- Given a sequence {a_n} satisfying the conditions:
    1) a₁ = 1
    2) a₂ = 1/2
    3) 2/a_{n+1} = 1/a_n + 1/a_{n+2} for all n ∈ ℕ*
    Prove that a₂₀₁₅ = 1/2015 -/
theorem sequence_2015th_term (a : ℕ → ℚ) 
  (h1 : a 1 = 1)
  (h2 : a 2 = 1/2)
  (h3 : ∀ n : ℕ, n ≥ 1 → 2 / (a (n + 1)) = 1 / (a n) + 1 / (a (n + 2))) :
  a 2015 = 1 / 2015 := by
  sorry

end sequence_2015th_term_l2378_237825


namespace unique_box_filling_l2378_237804

/-- Represents a rectangular parallelepiped with integer dimensions -/
structure Brick where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a brick -/
def Brick.volume (b : Brick) : ℕ := b.length * b.width * b.height

/-- The box to be filled -/
def box : Brick := ⟨10, 11, 14⟩

/-- The first type of brick -/
def brickA : Brick := ⟨2, 5, 8⟩

/-- The second type of brick -/
def brickB : Brick := ⟨2, 3, 7⟩

/-- Theorem stating that the only way to fill the box is with 14 bricks of type A and 10 of type B -/
theorem unique_box_filling :
  ∀ (x y : ℕ), 
    x * brickA.volume + y * brickB.volume = box.volume → 
    (x = 14 ∧ y = 10) := by sorry

end unique_box_filling_l2378_237804


namespace propositions_truth_l2378_237895

theorem propositions_truth : 
  (∀ a b : ℝ, a > 1 → b > 1 → a * b > 1) ∧ 
  (∃ a b c : ℝ, b = Real.sqrt (a * c) ∧ ¬(∃ r : ℝ, b = a * r ∧ c = b * r)) ∧
  (∃ a b c : ℝ, (∃ r : ℝ, b = a * r ∧ c = b * r) ∧ b ≠ Real.sqrt (a * c)) :=
by sorry


end propositions_truth_l2378_237895


namespace geometric_sequence_12th_term_l2378_237874

/-- A geometric sequence is defined by its first term and common ratio -/
def GeometricSequence (a₁ : ℝ) (r : ℝ) := fun n : ℕ => a₁ * r ^ (n - 1)

/-- The nth term of a geometric sequence -/
def nthTerm (seq : ℕ → ℝ) (n : ℕ) : ℝ := seq n

theorem geometric_sequence_12th_term
  (seq : ℕ → ℝ)
  (h_geometric : ∃ a₁ r, seq = GeometricSequence a₁ r)
  (h_4th : nthTerm seq 4 = 4)
  (h_7th : nthTerm seq 7 = 32) :
  nthTerm seq 12 = 1024 := by
  sorry


end geometric_sequence_12th_term_l2378_237874


namespace fractional_equation_solution_l2378_237856

theorem fractional_equation_solution (m : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → ((x + m) / (x - 2) + 1 / (2 - x) = 3)) →
  ((2 + m) / (2 - 2) + 1 / (2 - 2) = 3) →
  m = -1 := by
sorry

end fractional_equation_solution_l2378_237856


namespace work_days_calculation_l2378_237832

/-- Represents the number of days worked by each person -/
structure WorkDays where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The main theorem statement -/
theorem work_days_calculation 
  (work_days : WorkDays)
  (daily_wages : DailyWages)
  (total_earning : ℕ)
  (h1 : work_days.a = 6)
  (h2 : work_days.c = 4)
  (h3 : daily_wages.a * 4 = daily_wages.b * 3)
  (h4 : daily_wages.b * 5 = daily_wages.c * 4)
  (h5 : daily_wages.c = 95)
  (h6 : work_days.a * daily_wages.a + work_days.b * daily_wages.b + work_days.c * daily_wages.c = total_earning)
  (h7 : total_earning = 1406)
  : work_days.b = 9 := by
  sorry


end work_days_calculation_l2378_237832


namespace fireflies_problem_l2378_237898

theorem fireflies_problem (initial : ℕ) : 
  (initial + 8 - 2 = 9) → initial = 3 := by
  sorry

end fireflies_problem_l2378_237898


namespace point_order_l2378_237844

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 12

theorem point_order (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-1) = y₁)
  (h₂ : f (-3) = y₂)
  (h₃ : f 2 = y₃) :
  y₃ > y₂ ∧ y₂ > y₁ := by
  sorry

end point_order_l2378_237844


namespace sphere_surface_area_in_dihedral_angle_l2378_237802

/-- The surface area of a part of a sphere inside a dihedral angle -/
theorem sphere_surface_area_in_dihedral_angle 
  (R a α : ℝ) 
  (h_positive_R : R > 0)
  (h_positive_a : a > 0)
  (h_a_lt_R : a < R)
  (h_angle_range : 0 < α ∧ α < π) :
  let surface_area := 
    2 * R^2 * Real.arccos ((R * Real.cos α) / Real.sqrt (R^2 - a^2 * Real.sin α^2)) - 
    2 * R * a * Real.sin α * Real.arccos ((a * Real.cos α) / Real.sqrt (R^2 - a^2 * Real.sin α^2))
  surface_area > 0 ∧ surface_area < 4 * π * R^2 := by
  sorry

end sphere_surface_area_in_dihedral_angle_l2378_237802


namespace smaller_root_equation_l2378_237847

theorem smaller_root_equation (x : ℚ) : 
  let equation := (x - 3/4) * (x - 3/4) + (x - 3/4) * (x - 1/2) = 0
  let smaller_root := 5/8
  (equation ∧ x = smaller_root) ∨ 
  (equation ∧ x ≠ smaller_root ∧ x > smaller_root) :=
by sorry

end smaller_root_equation_l2378_237847


namespace expression_evaluation_l2378_237812

theorem expression_evaluation (m n : ℤ) (h1 : m = 2) (h2 : n = 1) : 
  (2 * m^2 - 3 * m * n + 8) - (5 * m * n - 4 * m^2 + 8) = 8 := by
  sorry

end expression_evaluation_l2378_237812


namespace households_with_bike_only_l2378_237827

/-- Proves that the number of households with only a bike is 35 -/
theorem households_with_bike_only
  (total : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (with_car : ℕ)
  (h_total : total = 90)
  (h_neither : neither = 11)
  (h_both : both = 16)
  (h_with_car : with_car = 44) :
  total - neither - (with_car - both) - both = 35 :=
by sorry

end households_with_bike_only_l2378_237827


namespace regular_triangular_pyramid_volume_l2378_237872

/-- A regular triangular pyramid -/
structure RegularTriangularPyramid where
  /-- The dihedral angle between two adjacent faces -/
  α : Real
  /-- The distance from the center of the base to an edge of the lateral face -/
  d : Real

/-- The volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : Real :=
  (9 * Real.tan p.α ^ 3) / (4 * Real.sqrt (3 * Real.tan p.α ^ 2 - 1))

theorem regular_triangular_pyramid_volume 
  (p : RegularTriangularPyramid) 
  (h1 : p.d = 1) 
  : volume p = (9 * Real.tan p.α ^ 3) / (4 * Real.sqrt (3 * Real.tan p.α ^ 2 - 1)) := by
  sorry

end regular_triangular_pyramid_volume_l2378_237872


namespace box_cubes_count_l2378_237889

/-- The minimum number of cubes required to build a box -/
def min_cubes (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height) / cube_volume

/-- Theorem: The minimum number of 3 cubic cm cubes required to build a box
    with dimensions 12 cm × 16 cm × 6 cm is 384. -/
theorem box_cubes_count :
  min_cubes 12 16 6 3 = 384 := by
  sorry

end box_cubes_count_l2378_237889


namespace geometric_sequence_sum_l2378_237852

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 * a 5 * a 6 = 8 →
  a 2 = 1 →
  a 2 + a 5 + a 8 + a 11 = 15 := by
sorry

end geometric_sequence_sum_l2378_237852


namespace lemonade_sale_duration_l2378_237818

/-- 
Given that Stanley sells 4 cups of lemonade per hour and Carl sells 7 cups per hour,
prove that they sold lemonade for 3 hours if Carl sold 9 more cups than Stanley.
-/
theorem lemonade_sale_duration : ∃ h : ℕ, h > 0 ∧ 7 * h = 4 * h + 9 := by
  sorry

end lemonade_sale_duration_l2378_237818


namespace sum_of_square_root_differences_l2378_237811

theorem sum_of_square_root_differences (S : ℝ) : 
  S = 1 / (4 - Real.sqrt 9) - 1 / (Real.sqrt 9 - Real.sqrt 8) + 
      1 / (Real.sqrt 8 - Real.sqrt 7) - 1 / (Real.sqrt 7 - Real.sqrt 6) + 
      1 / (Real.sqrt 6 - 3) → 
  S = 7 := by
sorry

end sum_of_square_root_differences_l2378_237811


namespace rabbits_in_park_l2378_237873

theorem rabbits_in_park (cage_rabbits : ℕ) (park_rabbits : ℕ) : 
  cage_rabbits = 13 →
  cage_rabbits + 7 = park_rabbits / 3 →
  park_rabbits = 60 := by
  sorry

end rabbits_in_park_l2378_237873


namespace cauliflower_earnings_l2378_237861

/-- Earnings from farmers' market --/
structure MarketEarnings where
  total : ℕ
  broccoli : ℕ
  carrots : ℕ
  spinach : ℕ
  cauliflower : ℕ

/-- Conditions for the farmers' market earnings --/
def validMarketEarnings (e : MarketEarnings) : Prop :=
  e.total = 380 ∧
  e.broccoli = 57 ∧
  e.carrots = 2 * e.broccoli ∧
  e.spinach = (e.carrots / 2) + 16 ∧
  e.total = e.broccoli + e.carrots + e.spinach + e.cauliflower

theorem cauliflower_earnings (e : MarketEarnings) (h : validMarketEarnings e) :
  e.cauliflower = 136 := by
  sorry

end cauliflower_earnings_l2378_237861


namespace cube_property_l2378_237814

theorem cube_property : ∃! (n : ℕ), n > 0 ∧ ∃ (k : ℕ), n^3 + 2*n^2 + 9*n + 8 = k^3 := by
  sorry

end cube_property_l2378_237814


namespace candy_left_l2378_237828

/-- Represents the number of candy pieces Debby has -/
def candy_count : ℕ := 12

/-- Represents the number of candy pieces Debby ate -/
def eaten_candy : ℕ := 9

/-- Theorem stating how many pieces of candy Debby has left -/
theorem candy_left : candy_count - eaten_candy = 3 := by sorry

end candy_left_l2378_237828


namespace product_comparison_l2378_237834

theorem product_comparison (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c := by
  sorry

end product_comparison_l2378_237834


namespace divide_forty_five_by_point_zero_five_l2378_237894

theorem divide_forty_five_by_point_zero_five : 45 / 0.05 = 900 := by
  sorry

end divide_forty_five_by_point_zero_five_l2378_237894


namespace inequality_solution_range_l2378_237870

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x < 1 ↔ 2*a*x + 3*x > 2*a + 3) → a < -3/2 :=
by sorry

end inequality_solution_range_l2378_237870


namespace five_Z_three_equals_twelve_l2378_237859

-- Define the Z operation
def Z (a b : ℝ) : ℝ := 3 * (a - b)^2

-- Theorem statement
theorem five_Z_three_equals_twelve : Z 5 3 = 12 := by
  sorry

end five_Z_three_equals_twelve_l2378_237859


namespace rational_a_condition_l2378_237855

theorem rational_a_condition (m n : ℤ) : 
  ∃ (a : ℚ), a = (m^4 + n^4 + m^2*n^2) / (4*m^2*n^2) :=
by sorry

end rational_a_condition_l2378_237855


namespace converse_proposition_l2378_237809

theorem converse_proposition : ∀ x : ℝ, (1 / (x - 1) ≥ 3) → (x ≤ 4 / 3) := by sorry

end converse_proposition_l2378_237809


namespace mary_screw_ratio_l2378_237841

/-- The number of screws Mary initially has -/
def initial_screws : ℕ := 8

/-- The number of sections Mary needs to split the screws into -/
def num_sections : ℕ := 4

/-- The number of screws needed in each section -/
def screws_per_section : ℕ := 6

/-- The ratio of screws Mary needs to buy to the screws she initially has -/
def screw_ratio : ℚ := 2

theorem mary_screw_ratio : 
  (num_sections * screws_per_section - initial_screws) / initial_screws = screw_ratio := by
  sorry

end mary_screw_ratio_l2378_237841


namespace pear_sales_l2378_237830

theorem pear_sales (morning_sales afternoon_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  morning_sales + afternoon_sales = 360 →
  afternoon_sales = 240 := by
sorry

end pear_sales_l2378_237830


namespace f_sum_2016_2015_l2378_237838

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_sum_2016_2015 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_even : is_even_function (fun x ↦ f (x + 1)))
  (h_f_1 : f 1 = 1) :
  f 2016 + f 2015 = -1 := by
  sorry

end f_sum_2016_2015_l2378_237838


namespace initial_number_of_persons_l2378_237896

theorem initial_number_of_persons
  (average_weight_increase : ℝ)
  (weight_difference : ℝ)
  (h1 : average_weight_increase = 2.5)
  (h2 : weight_difference = 20)
  (h3 : average_weight_increase * (initial_persons : ℝ) = weight_difference) :
  initial_persons = 8 := by
sorry

end initial_number_of_persons_l2378_237896


namespace prime_product_divisible_by_four_l2378_237897

theorem prime_product_divisible_by_four (p q : ℕ) : 
  Prime p → Prime q → Prime (p * q + 1) → 
  4 ∣ ((2 * p + q) * (p + 2 * q)) := by
sorry

end prime_product_divisible_by_four_l2378_237897


namespace rectangle_area_l2378_237801

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 1225)
  (h2 : rectangle_breadth = 10)
  : ∃ (circle_radius : ℝ) (rectangle_length : ℝ),
    circle_radius ^ 2 = square_area ∧
    rectangle_length = (2 / 5) * circle_radius ∧
    rectangle_length * rectangle_breadth = 140 := by
  sorry

end rectangle_area_l2378_237801


namespace quadratic_minimum_l2378_237835

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

theorem quadratic_minimum (p q : ℝ) :
  (∀ x, f p q x ≥ f p q q) ∧ 
  (f p q q = (p + q)^2) →
  ((p = 0 ∧ q = 0) ∨ (p = -1 ∧ q = 1/2)) := by
sorry

end quadratic_minimum_l2378_237835


namespace y_minus_x_equals_one_tenth_l2378_237853

-- Define the rounding function to the tenths place
def roundToTenths (x : ℚ) : ℚ := ⌊x * 10 + 1/2⌋ / 10

-- Define the given values
def a : ℚ := 545/100
def b : ℚ := 295/100
def c : ℚ := 374/100

-- Define x as the sum of a, b, and c rounded to tenths
def x : ℚ := roundToTenths (a + b + c)

-- Define y as the sum of a, b, and c individually rounded to tenths
def y : ℚ := roundToTenths a + roundToTenths b + roundToTenths c

-- State the theorem
theorem y_minus_x_equals_one_tenth : y - x = 1/10 := by sorry

end y_minus_x_equals_one_tenth_l2378_237853


namespace right_triangle_consecutive_sides_l2378_237890

theorem right_triangle_consecutive_sides (a c : ℕ) (b : ℝ) : 
  c = a + 1 → -- c and a are consecutive integers
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  b^2 = c + a := by
sorry

end right_triangle_consecutive_sides_l2378_237890


namespace jake_and_kendra_weight_l2378_237839

/-- Calculates the combined weight of Jake and Kendra given Jake's current weight and the condition about their weight relation after Jake loses 8 pounds. -/
def combinedWeight (jakeWeight : ℕ) : ℕ :=
  let kendraWeight := (jakeWeight - 8) / 2
  jakeWeight + kendraWeight

/-- Theorem stating that given Jake's current weight of 196 pounds and the condition about their weight relation, the combined weight of Jake and Kendra is 290 pounds. -/
theorem jake_and_kendra_weight : combinedWeight 196 = 290 := by
  sorry

#eval combinedWeight 196

end jake_and_kendra_weight_l2378_237839


namespace initial_money_calculation_l2378_237833

theorem initial_money_calculation (initial_amount : ℚ) : 
  (initial_amount * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 500) → 
  initial_amount = 1250 := by
sorry

end initial_money_calculation_l2378_237833


namespace compute_expression_l2378_237893

theorem compute_expression : 9 * (-5) - (7 * -2) + (8 * -6) = -79 := by
  sorry

end compute_expression_l2378_237893


namespace downstream_distance_l2378_237882

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (travel_time : ℝ)
  (h1 : boat_speed = 22)
  (h2 : stream_speed = 5)
  (h3 : travel_time = 8) :
  boat_speed + stream_speed * travel_time = 216 :=
by
  sorry

#check downstream_distance

end downstream_distance_l2378_237882


namespace complex_equation_solution_l2378_237820

theorem complex_equation_solution (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) : z = 2 - I := by
  sorry

end complex_equation_solution_l2378_237820


namespace concentric_circles_ratio_l2378_237865

theorem concentric_circles_ratio (a b : ℝ) (h : a > 0) (h' : b > 0) 
  (h_area : π * b^2 - π * a^2 = 4 * (π * a^2)) : 
  a / b = 1 / Real.sqrt 5 := by
sorry

end concentric_circles_ratio_l2378_237865


namespace classmate_reading_comprehensive_only_classmate_reading_comprehensive_l2378_237823

/-- Represents a survey activity -/
inductive SurveyActivity
| SocketLifespan
| TreePlantingSurvival
| ClassmateReading
| DocumentaryViewership

/-- Determines if a survey activity is suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (activity : SurveyActivity) : Prop :=
  match activity with
  | SurveyActivity.ClassmateReading => true
  | _ => false

/-- Theorem stating that the classmate reading survey is suitable for a comprehensive survey -/
theorem classmate_reading_comprehensive :
  isSuitableForComprehensiveSurvey SurveyActivity.ClassmateReading :=
by sorry

/-- Theorem stating that the classmate reading survey is the only suitable activity for a comprehensive survey -/
theorem only_classmate_reading_comprehensive (activity : SurveyActivity) :
  isSuitableForComprehensiveSurvey activity ↔ activity = SurveyActivity.ClassmateReading :=
by sorry

end classmate_reading_comprehensive_only_classmate_reading_comprehensive_l2378_237823


namespace birds_in_tree_l2378_237869

theorem birds_in_tree (initial_birds new_birds : ℕ) 
  (h1 : initial_birds = 14) 
  (h2 : new_birds = 21) : 
  initial_birds + new_birds = 35 :=
by sorry

end birds_in_tree_l2378_237869


namespace intersection_A_complement_B_equals_A_l2378_237842

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- Theorem statement
theorem intersection_A_complement_B_equals_A : A ∩ (U \ B) = A := by sorry

end intersection_A_complement_B_equals_A_l2378_237842


namespace fourth_rectangle_area_l2378_237881

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a large rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  large : Rectangle
  small1 : Rectangle
  small2 : Rectangle
  small3 : Rectangle
  small4 : Rectangle

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: If a rectangle is divided into four smaller rectangles, and three of them have
    areas 20, 12, and 16, then the fourth rectangle has an area of 16 -/
theorem fourth_rectangle_area
  (dr : DividedRectangle)
  (h1 : area dr.small1 = 20)
  (h2 : area dr.small2 = 12)
  (h3 : area dr.small3 = 16)
  (h_sum : area dr.large = area dr.small1 + area dr.small2 + area dr.small3 + area dr.small4)
  : area dr.small4 = 16 := by
  sorry

end fourth_rectangle_area_l2378_237881


namespace increasing_f_implies_a_range_l2378_237879

def f (a x : ℝ) : ℝ := x^2 + 2*(a-2)*x + 5

theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 4 ≤ x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) →
  a ∈ Set.Ici (-2) :=
by sorry

end increasing_f_implies_a_range_l2378_237879


namespace largest_c_for_3_in_range_l2378_237836

def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

theorem largest_c_for_3_in_range : 
  (∃ (c : ℝ), ∀ (c' : ℝ), 
    (∃ (x : ℝ), f c' x = 3) → c' ≤ c ∧ 
    (∃ (x : ℝ), f c x = 3) ∧
    c = 12) := by sorry

end largest_c_for_3_in_range_l2378_237836


namespace gladys_age_ratio_l2378_237807

def gladys_age : ℕ := 30

def billy_age : ℕ := gladys_age / 3

def lucas_age : ℕ := 8 - 3

def sum_billy_lucas : ℕ := billy_age + lucas_age

theorem gladys_age_ratio : 
  gladys_age / sum_billy_lucas = 2 :=
by sorry

end gladys_age_ratio_l2378_237807


namespace sum_bounds_l2378_237806

theorem sum_bounds (a b c d e : ℝ) :
  0 < (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) ∧
  (a / (a + b) + b / (b + c) + c / (c + d) + d / (d + e) + e / (e + a)) < 5 := by
  sorry

end sum_bounds_l2378_237806


namespace school_fair_revenue_l2378_237875

/-- Calculates the total revenue from sales at a school fair -/
theorem school_fair_revenue (chips_sold : ℕ) (chips_price : ℚ)
  (hot_dogs_sold : ℕ) (hot_dogs_price : ℚ)
  (drinks_sold : ℕ) (drinks_price : ℚ) :
  chips_sold = 27 →
  chips_price = 3/2 →
  hot_dogs_sold = chips_sold - 8 →
  hot_dogs_price = 3 →
  drinks_sold = hot_dogs_sold + 12 →
  drinks_price = 2 →
  chips_sold * chips_price + hot_dogs_sold * hot_dogs_price + drinks_sold * drinks_price = 159.5 := by
  sorry

#eval (27 : ℕ) * (3/2 : ℚ) + (27 - 8 : ℕ) * (3 : ℚ) + ((27 - 8 : ℕ) + 12) * (2 : ℚ)

end school_fair_revenue_l2378_237875


namespace log2_odd_and_increasing_l2378_237864

-- Define the logarithm base 2 function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log2_odd_and_increasing :
  (∀ x > 0, log2 (-x) = -log2 x) ∧
  (∀ x y, 0 ≤ x → x ≤ y → log2 x ≤ log2 y) :=
by sorry

end log2_odd_and_increasing_l2378_237864


namespace smallest_factor_difference_l2378_237876

theorem smallest_factor_difference (n : ℕ) (hn : n = 2310) :
  ∃ (a b : ℕ), a * b = n ∧ 
    (∀ (x y : ℕ), x * y = n → x ≤ y → y - x ≥ (b - a)) ∧
    b - a = 13 :=
  sorry

end smallest_factor_difference_l2378_237876


namespace circle_radius_equality_l2378_237885

theorem circle_radius_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 37) (h₂ : r₂ = 23) :
  ∃ r : ℝ, r^2 = (r₁^2 - r₂^2) ∧ r = 2 * Real.sqrt 210 := by
  sorry

end circle_radius_equality_l2378_237885


namespace nesbitts_inequality_l2378_237878

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 := by
  sorry

end nesbitts_inequality_l2378_237878


namespace triangle_properties_l2378_237848

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  4 * a = Real.sqrt 5 * c ∧
  Real.cos C = 3 / 5 →
  Real.sin A = Real.sqrt 5 / 5 ∧
  (b = 11 → a = 5) ∧
  (b = 11 → Real.cos (2 * A + C) = -7 / 25) := by
sorry


end triangle_properties_l2378_237848


namespace marcus_cookies_count_l2378_237829

/-- The number of peanut butter cookies Marcus brought to the bake sale -/
def marcus_peanut_butter_cookies : ℕ := 30

/-- The number of peanut butter cookies Jenny brought to the bake sale -/
def jenny_peanut_butter_cookies : ℕ := 40

/-- The total number of non-peanut butter cookies at the bake sale -/
def total_non_peanut_butter_cookies : ℕ := 70

/-- The probability of picking a peanut butter cookie -/
def peanut_butter_probability : ℚ := 1/2

theorem marcus_cookies_count :
  marcus_peanut_butter_cookies = 30 ∧
  jenny_peanut_butter_cookies + marcus_peanut_butter_cookies = total_non_peanut_butter_cookies ∧
  (jenny_peanut_butter_cookies + marcus_peanut_butter_cookies : ℚ) /
    (jenny_peanut_butter_cookies + marcus_peanut_butter_cookies + total_non_peanut_butter_cookies) = peanut_butter_probability :=
by sorry

end marcus_cookies_count_l2378_237829


namespace quadratic_inequality_solution_l2378_237816

theorem quadratic_inequality_solution (t a : ℝ) : 
  (∀ x, tx^2 - 6*x + t^2 < 0 ↔ x ∈ Set.Ioi 1 ∪ Set.Iic a) →
  (t*a^2 - 6*a + t^2 = 0 ∧ t*1^2 - 6*1 + t^2 = 0) →
  t < 0 →
  a = -3 := by sorry

end quadratic_inequality_solution_l2378_237816


namespace no_prime_divisible_by_39_l2378_237822

theorem no_prime_divisible_by_39 : ∀ p : ℕ, Prime p → ¬(39 ∣ p) := by
  sorry

end no_prime_divisible_by_39_l2378_237822


namespace arithmetic_sequence_sum_l2378_237845

theorem arithmetic_sequence_sum (a₁ a₂ a₃ a₆ : ℕ) (h₁ : a₁ = 5) (h₂ : a₂ = 12) (h₃ : a₃ = 19) (h₆ : a₆ = 40) :
  let d := a₂ - a₁
  let a₄ := a₃ + d
  let a₅ := a₄ + d
  a₄ + a₅ = 59 := by sorry

end arithmetic_sequence_sum_l2378_237845


namespace problem_solution_l2378_237850

theorem problem_solution (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 := by
  sorry

end problem_solution_l2378_237850


namespace eight_sided_die_probability_l2378_237891

/-- Represents the number of sides on the die -/
def sides : ℕ := 8

/-- Represents the event where the first roll is greater than or equal to the second roll -/
def favorable_outcomes (s : ℕ) : ℕ := (s * (s + 1)) / 2

/-- The probability of the first roll being greater than or equal to the second roll -/
def probability (s : ℕ) : ℚ := (favorable_outcomes s) / (s^2 : ℚ)

/-- Theorem stating that for an 8-sided die, the probability of the first roll being 
    greater than or equal to the second roll is 9/16 -/
theorem eight_sided_die_probability : probability sides = 9/16 := by
  sorry

end eight_sided_die_probability_l2378_237891


namespace no_integer_solutions_l2378_237860

theorem no_integer_solutions (n : ℤ) (s : ℕ) (h_s : Odd s) :
  ¬ ∃ x : ℤ, x^2 - 16*n*x + 7^s = 0 :=
by sorry

end no_integer_solutions_l2378_237860


namespace line_perpendicular_to_plane_l2378_237868

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular n β) : 
  perpendicular m β :=
sorry

end line_perpendicular_to_plane_l2378_237868


namespace worker_travel_time_l2378_237803

/-- Proves that if a worker walking at 5/6 of her normal speed arrives 12 minutes later than usual, her usual travel time is 60 minutes. -/
theorem worker_travel_time (normal_speed : ℝ) (normal_time : ℝ) : 
  normal_speed * normal_time = (5/6 * normal_speed) * (normal_time + 12) → 
  normal_time = 60 := by
  sorry

end worker_travel_time_l2378_237803


namespace vector_sum_necessary_not_sufficient_l2378_237887

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def form_triangle (a b c : V) : Prop := sorry

theorem vector_sum_necessary_not_sufficient (a b c : V) :
  (form_triangle a b c → a + b + c = 0) ∧
  ¬(a + b + c = 0 → form_triangle a b c) :=
by sorry

end vector_sum_necessary_not_sufficient_l2378_237887


namespace player_A_performance_l2378_237819

/-- Represents a basketball player's shooting performance -/
structure Player where
  shotProbability : ℝ
  roundsTotal : ℕ
  shotsPerRound : ℕ

/-- Calculates the probability of passing a single round -/
def passProbability (p : Player) : ℝ :=
  1 - (1 - p.shotProbability) ^ p.shotsPerRound

/-- Calculates the expected number of passed rounds -/
def expectedPassedRounds (p : Player) : ℝ :=
  p.roundsTotal * passProbability p

/-- Theorem stating the probability of passing a round and expected passed rounds for player A -/
theorem player_A_performance : 
  let playerA : Player := { shotProbability := 0.6, roundsTotal := 5, shotsPerRound := 2 }
  passProbability playerA = 0.84 ∧ expectedPassedRounds playerA = 4.2 := by
  sorry


end player_A_performance_l2378_237819


namespace expected_final_set_size_l2378_237863

/-- The set of elements Marisa is working with -/
def S : Finset Nat := Finset.range 8

/-- The initial number of subsets in Marisa's collection -/
def initial_subsets : Nat := 2^8 - 1

/-- The number of steps in Marisa's process -/
def num_steps : Nat := 2^8 - 2

/-- The probability of an element being in a randomly chosen subset -/
def prob_in_subset : ℚ := 128 / 255

/-- The expected size of the final set in Marisa's subset collection process -/
theorem expected_final_set_size :
  (S.card : ℚ) * prob_in_subset = 1024 / 255 := by sorry

end expected_final_set_size_l2378_237863


namespace tv_screen_area_l2378_237857

theorem tv_screen_area : 
  let trapezoid_short_base : ℝ := 3
  let trapezoid_long_base : ℝ := 5
  let trapezoid_height : ℝ := 2
  let triangle_base : ℝ := trapezoid_long_base
  let triangle_height : ℝ := 4
  let trapezoid_area := (trapezoid_short_base + trapezoid_long_base) * trapezoid_height / 2
  let triangle_area := triangle_base * triangle_height / 2
  trapezoid_area + triangle_area = 18 := by
sorry

end tv_screen_area_l2378_237857


namespace time_for_b_alone_l2378_237851

/-- Given that:
  1. It takes 'a' hours for A and B to complete the work together.
  2. It takes 'b' hours for A to complete the work alone.
  Prove that the time it takes B alone to complete the work is ab / (b - a) hours. -/
theorem time_for_b_alone (a b : ℝ) (h1 : a > 0) (h2 : b > a) : 
  (1 / a + 1 / (a * b / (b - a)) = 1) := by
sorry

end time_for_b_alone_l2378_237851


namespace parabola_y_intercepts_l2378_237858

/-- The number of y-intercepts for the parabola x = 3y^2 - 6y + 3 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 6 * y + 3
  ∃! y : ℝ, f y = 0 :=
by sorry

end parabola_y_intercepts_l2378_237858


namespace problem_one_problem_two_l2378_237837

-- Problem 1
theorem problem_one : -9 + 5 - (-12) + (-3) = 5 := by
  sorry

-- Problem 2
theorem problem_two : -(1.5) - (-4.25) + 3.75 - 8.5 = -2 := by
  sorry

end problem_one_problem_two_l2378_237837
