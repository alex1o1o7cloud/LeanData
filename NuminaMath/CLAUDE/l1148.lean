import Mathlib

namespace NUMINAMATH_CALUDE_function_equation_solution_l1148_114820

theorem function_equation_solution (a b : ℚ) :
  ∃ (f : ℚ → ℚ), (∀ x y : ℚ, f (x + a + f y) = f (x + b) + y) →
  ∃ A : ℚ, ∀ x : ℚ, f x = A * x + (a - b) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1148_114820


namespace NUMINAMATH_CALUDE_find_b_l1148_114883

-- Define the relationship between a and b
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^2 * Real.sqrt b = k

-- Define the theorem
theorem find_b (a b : ℝ) (h1 : inverse_relation a b) (h2 : a = 2 ∧ b = 81) (h3 : a * b = 48) :
  b = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l1148_114883


namespace NUMINAMATH_CALUDE_set_problem_l1148_114807

def U : Set ℕ := {x | x ≤ 10}

theorem set_problem (A B : Set ℕ) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {4,5,6})
  (h4 : (U \ B) ∩ A = {2,3})
  (h5 : (U \ A) ∩ (U \ B) = {7,8}) :
  A = {2,3,4,5,6} ∧ B = {4,5,6,9,10} := by
  sorry

end NUMINAMATH_CALUDE_set_problem_l1148_114807


namespace NUMINAMATH_CALUDE_exactly_two_b_values_l1148_114863

theorem exactly_two_b_values : 
  ∃! (s : Finset ℤ), 
    (∀ b ∈ s, ∃! (t : Finset ℤ), 
      (∀ x ∈ t, x^2 + b*x + 6 ≤ 0) ∧ 
      (∀ x ∉ t, x^2 + b*x + 6 > 0) ∧ 
      t.card = 3) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_b_values_l1148_114863


namespace NUMINAMATH_CALUDE_yoongi_hoseok_age_sum_l1148_114803

/-- Given the ages of Yoongi's aunt, the age difference between Yoongi and his aunt,
    and the age difference between Yoongi and Hoseok, prove that the sum of
    Yoongi and Hoseok's ages is 26 years. -/
theorem yoongi_hoseok_age_sum :
  ∀ (aunt_age : ℕ) (yoongi_aunt_diff : ℕ) (yoongi_hoseok_diff : ℕ),
  aunt_age = 38 →
  yoongi_aunt_diff = 23 →
  yoongi_hoseok_diff = 4 →
  (aunt_age - yoongi_aunt_diff) + (aunt_age - yoongi_aunt_diff - yoongi_hoseok_diff) = 26 :=
by sorry

end NUMINAMATH_CALUDE_yoongi_hoseok_age_sum_l1148_114803


namespace NUMINAMATH_CALUDE_sean_initial_blocks_l1148_114848

/-- The number of blocks Sean had initially -/
def initial_blocks : ℕ := sorry

/-- The number of blocks eaten by the hippopotamus -/
def blocks_eaten : ℕ := 29

/-- The number of blocks remaining after the hippopotamus ate some -/
def blocks_remaining : ℕ := 26

/-- Theorem stating that Sean initially had 55 blocks -/
theorem sean_initial_blocks : initial_blocks = 55 := by sorry

end NUMINAMATH_CALUDE_sean_initial_blocks_l1148_114848


namespace NUMINAMATH_CALUDE_opposite_of_one_fourth_l1148_114809

theorem opposite_of_one_fourth : -(1 / 4 : ℚ) = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_fourth_l1148_114809


namespace NUMINAMATH_CALUDE_product_equals_expansion_l1148_114850

-- Define the binomials
def binomial1 (x : ℝ) : ℝ := 4 * x + 3
def binomial2 (x : ℝ) : ℝ := 2 * x - 7

-- Define the product using the distributive property
def product (x : ℝ) : ℝ := binomial1 x * binomial2 x

-- Theorem stating that the product equals the expanded form
theorem product_equals_expansion (x : ℝ) : 
  product x = 8 * x^2 - 22 * x - 21 := by sorry

end NUMINAMATH_CALUDE_product_equals_expansion_l1148_114850


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l1148_114800

/-- Proves that the minimum bailing rate is 13 gallons per minute -/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (water_intake_rate : ℝ)
  (boat_capacity : ℝ)
  (rowing_speed : ℝ)
  (h1 : distance_to_shore = 3)
  (h2 : water_intake_rate = 15)
  (h3 : boat_capacity = 60)
  (h4 : rowing_speed = 6)
  : ∃ (bailing_rate : ℝ), 
    bailing_rate = 13 ∧ 
    bailing_rate * (distance_to_shore / rowing_speed * 60) ≥ 
    water_intake_rate * (distance_to_shore / rowing_speed * 60) - boat_capacity ∧
    ∀ (r : ℝ), r < bailing_rate → 
      r * (distance_to_shore / rowing_speed * 60) < 
      water_intake_rate * (distance_to_shore / rowing_speed * 60) - boat_capacity :=
by sorry


end NUMINAMATH_CALUDE_minimum_bailing_rate_l1148_114800


namespace NUMINAMATH_CALUDE_least_people_second_caterer_cheaper_l1148_114828

def first_caterer_cost (people : ℕ) : ℕ := 100 + 15 * people
def second_caterer_cost (people : ℕ) : ℕ := 200 + 12 * people

theorem least_people_second_caterer_cheaper :
  (∀ n : ℕ, n < 34 → first_caterer_cost n ≤ second_caterer_cost n) ∧
  (second_caterer_cost 34 < first_caterer_cost 34) := by
  sorry

end NUMINAMATH_CALUDE_least_people_second_caterer_cheaper_l1148_114828


namespace NUMINAMATH_CALUDE_ellen_smoothie_strawberries_l1148_114806

/-- The amount of strawberries used in Ellen's smoothie recipe. -/
def strawberries : ℝ := 0.5 - (0.1 + 0.2)

/-- Theorem stating that Ellen used 0.2 cups of strawberries in her smoothie. -/
theorem ellen_smoothie_strawberries :
  strawberries = 0.2 := by sorry

end NUMINAMATH_CALUDE_ellen_smoothie_strawberries_l1148_114806


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1148_114815

/-- Proves that given a hyperbola with specific conditions, its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧  -- hyperbola equation
  (∃ (x y : ℝ), y = Real.sqrt 3 * x) ∧    -- asymptote condition
  (∃ (x y : ℝ), y^2 = 16*x ∧ x^2/a^2 + y^2/b^2 = 1) -- focus on directrix condition
  →
  a^2 = 4 ∧ b^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1148_114815


namespace NUMINAMATH_CALUDE_remainder_theorem_l1148_114860

-- Define the polynomial and its divisor
def p (x : ℝ) : ℝ := 3*x^7 + 2*x^5 - 5*x^3 + x^2 - 9
def d (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the remainder
def r (x : ℝ) : ℝ := 14*x - 16

-- Theorem statement
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = d x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1148_114860


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l1148_114829

theorem halfway_between_one_eighth_and_one_third :
  (1 / 8 + 1 / 3) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l1148_114829


namespace NUMINAMATH_CALUDE_lcm_of_5_6_10_18_l1148_114830

theorem lcm_of_5_6_10_18 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 18)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_10_18_l1148_114830


namespace NUMINAMATH_CALUDE_collinear_vectors_l1148_114812

/-- Given vectors a, b, and c in ℝ², prove that if a - 2b is collinear with c, then k = 1 -/
theorem collinear_vectors (a b c : ℝ × ℝ) (h : a = (Real.sqrt 3, 1)) (h' : b = (0, -1)) 
    (h'' : c = (k, Real.sqrt 3)) (h''' : ∃ t : ℝ, a - 2 • b = t • c) : k = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l1148_114812


namespace NUMINAMATH_CALUDE_sequence_properties_l1148_114864

def sequence_a (n : ℕ) : ℤ := 2^n - n - 2

def sequence_c (n : ℕ) : ℤ := sequence_a n + n + 2

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequence_a (n + 1) = 2 * sequence_a n + n + 1) ∧
  (sequence_a 1 = -1) ∧
  (∀ n : ℕ, n > 0 → sequence_c (n + 1) = 2 * sequence_c n) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1148_114864


namespace NUMINAMATH_CALUDE_least_number_of_marbles_l1148_114825

def is_divisible_by_all (n : ℕ) : Prop :=
  ∀ i ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ), n % i = 0

theorem least_number_of_marbles :
  ∃ (n : ℕ), n > 0 ∧ is_divisible_by_all n ∧ ∀ m, 0 < m ∧ m < n → ¬is_divisible_by_all m :=
by
  use 840
  sorry

end NUMINAMATH_CALUDE_least_number_of_marbles_l1148_114825


namespace NUMINAMATH_CALUDE_triangle_vertices_l1148_114841

-- Define the lines
def d₁ (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def d₂ (x y : ℝ) : Prop := x + y - 4 = 0
def d₃ (x y : ℝ) : Prop := y = 2
def d₄ (x y : ℝ) : Prop := x - 4 * y + 3 = 0

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (5, 2)

-- Define what it means for a line to be a median
def is_median (line : (ℝ → ℝ → Prop)) (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

-- Define what it means for a line to be an altitude
def is_altitude (line : (ℝ → ℝ → Prop)) (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

theorem triangle_vertices : 
  is_median d₁ (A, B, C) ∧ 
  is_median d₂ (A, B, C) ∧ 
  is_median d₃ (A, B, C) ∧ 
  is_altitude d₄ (A, B, C) → 
  (A = (1, 0) ∧ B = (0, 4) ∧ C = (5, 2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_vertices_l1148_114841


namespace NUMINAMATH_CALUDE_prom_ticket_cost_l1148_114886

def total_cost : ℝ := 836
def dinner_cost : ℝ := 120
def tip_percentage : ℝ := 0.30
def limo_cost_per_hour : ℝ := 80
def limo_rental_duration : ℝ := 6
def number_of_tickets : ℝ := 2

theorem prom_ticket_cost :
  let tip_cost := dinner_cost * tip_percentage
  let limo_total_cost := limo_cost_per_hour * limo_rental_duration
  let total_cost_without_tickets := dinner_cost + tip_cost + limo_total_cost
  let ticket_total_cost := total_cost - total_cost_without_tickets
  let ticket_cost := ticket_total_cost / number_of_tickets
  ticket_cost = 100 := by sorry

end NUMINAMATH_CALUDE_prom_ticket_cost_l1148_114886


namespace NUMINAMATH_CALUDE_odd_integer_sum_theorem_l1148_114885

/-- The sum of 60 non-consecutive, odd integers starting from -29 in increasing order -/
def oddIntegerSum : ℤ := 5340

/-- The first term of the sequence -/
def firstTerm : ℤ := -29

/-- The number of terms in the sequence -/
def numTerms : ℕ := 60

/-- The common difference between consecutive terms -/
def commonDiff : ℤ := 4

/-- The last term of the sequence -/
def lastTerm : ℤ := firstTerm + (numTerms - 1) * commonDiff

theorem odd_integer_sum_theorem :
  oddIntegerSum = (numTerms : ℤ) * (firstTerm + lastTerm) / 2 :=
sorry

end NUMINAMATH_CALUDE_odd_integer_sum_theorem_l1148_114885


namespace NUMINAMATH_CALUDE_urn_problem_l1148_114888

theorem urn_problem (N : ℕ) : 
  let urn1_red : ℕ := 5
  let urn1_yellow : ℕ := 8
  let urn2_red : ℕ := 18
  let urn2_yellow : ℕ := N
  let total1 : ℕ := urn1_red + urn1_yellow
  let total2 : ℕ := urn2_red + urn2_yellow
  let prob_same_color : ℚ := (urn1_red / total1) * (urn2_red / total2) + 
                             (urn1_yellow / total1) * (urn2_yellow / total2)
  prob_same_color = 62/100 → N = 59 := by
sorry


end NUMINAMATH_CALUDE_urn_problem_l1148_114888


namespace NUMINAMATH_CALUDE_operation_simplification_l1148_114856

theorem operation_simplification (x : ℚ) : 
  ((3 * x + 6) - 5 * x + 10) / 5 = -2/5 * x + 16/5 := by
  sorry

end NUMINAMATH_CALUDE_operation_simplification_l1148_114856


namespace NUMINAMATH_CALUDE_function_value_at_two_l1148_114817

theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2*x) : f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l1148_114817


namespace NUMINAMATH_CALUDE_jury_seating_arrangements_l1148_114872

/-- Represents the number of jury members -/
def n : ℕ := 12

/-- Represents the number of jury members excluding Nikolai Nikolaevich and the person whose seat he took -/
def m : ℕ := n - 2

/-- A function that calculates the number of distinct seating arrangements -/
def seating_arrangements (n : ℕ) : ℕ := 2^(n - 2)

/-- Theorem stating that the number of distinct seating arrangements for 12 jury members is 2^10 -/
theorem jury_seating_arrangements :
  seating_arrangements n = 2^m :=
by sorry

end NUMINAMATH_CALUDE_jury_seating_arrangements_l1148_114872


namespace NUMINAMATH_CALUDE_mrs_hilt_shopping_l1148_114831

/-- Mrs. Hilt's shopping problem -/
theorem mrs_hilt_shopping (pencil_cost candy_cost remaining_money : ℕ) 
  (h1 : pencil_cost = 20)
  (h2 : candy_cost = 5)
  (h3 : remaining_money = 18) :
  pencil_cost + candy_cost + remaining_money = 43 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_shopping_l1148_114831


namespace NUMINAMATH_CALUDE_rectangle_variability_l1148_114893

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the perimeter, area, and one side length
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)
def area (r : Rectangle) : ℝ := r.length * r.width
def oneSideLength (r : Rectangle) : ℝ := r.length

-- State the theorem
theorem rectangle_variability (fixedPerimeter : ℝ) (r : Rectangle) 
  (h : perimeter r = fixedPerimeter) :
  ∃ (r' : Rectangle), 
    perimeter r' = fixedPerimeter ∧ 
    area r' ≠ area r ∧
    oneSideLength r' ≠ oneSideLength r :=
sorry

end NUMINAMATH_CALUDE_rectangle_variability_l1148_114893


namespace NUMINAMATH_CALUDE_gardening_project_cost_correct_l1148_114897

def gardening_project_cost (rose_bushes : ℕ) (rose_bush_cost : ℕ) (fertilizer_cost : ℕ) 
  (gardener_hours : List ℕ) (gardener_rate : ℕ) (soil_volume : ℕ) (soil_cost : ℕ) : ℕ :=
  let bush_total := rose_bushes * rose_bush_cost
  let fertilizer_total := rose_bushes * fertilizer_cost
  let labor_total := (List.sum gardener_hours) * gardener_rate
  let soil_total := soil_volume * soil_cost
  bush_total + fertilizer_total + labor_total + soil_total

theorem gardening_project_cost_correct : 
  gardening_project_cost 20 150 25 [6, 5, 4, 7] 30 100 5 = 4660 := by
  sorry

end NUMINAMATH_CALUDE_gardening_project_cost_correct_l1148_114897


namespace NUMINAMATH_CALUDE_at_least_one_meets_standard_l1148_114804

theorem at_least_one_meets_standard (pA pB pC : ℝ) 
  (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_meets_standard_l1148_114804


namespace NUMINAMATH_CALUDE_michaels_regular_hours_l1148_114867

/-- Proves that given the conditions of Michael's work schedule and earnings,
    the number of regular hours worked before overtime is 40. -/
theorem michaels_regular_hours
  (regular_rate : ℝ)
  (total_earnings : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 7)
  (h2 : total_earnings = 320)
  (h3 : total_hours = 42.857142857142854)
  : ∃ (regular_hours : ℝ),
    regular_hours = 40 ∧
    regular_hours * regular_rate +
    (total_hours - regular_hours) * (2 * regular_rate) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_michaels_regular_hours_l1148_114867


namespace NUMINAMATH_CALUDE_martha_started_with_three_cards_l1148_114827

/-- The number of cards Martha started with -/
def initial_cards : ℕ := sorry

/-- The number of cards Martha received from Emily -/
def cards_from_emily : ℕ := 76

/-- The total number of cards Martha ended up with -/
def total_cards : ℕ := 79

/-- Theorem stating that Martha started with 3 cards -/
theorem martha_started_with_three_cards : 
  initial_cards = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_started_with_three_cards_l1148_114827


namespace NUMINAMATH_CALUDE_djibo_sister_age_djibo_sister_age_is_28_l1148_114857

theorem djibo_sister_age (djibo_current_age : ℕ) (sum_five_years_ago : ℕ) : ℕ :=
  let djibo_age_five_years_ago := djibo_current_age - 5
  let sister_age_five_years_ago := sum_five_years_ago - djibo_age_five_years_ago
  sister_age_five_years_ago + 5

/-- Given Djibo's current age and the sum of his and his sister's ages five years ago,
    prove that Djibo's sister's current age is 28. -/
theorem djibo_sister_age_is_28 :
  djibo_sister_age 17 35 = 28 := by
  sorry

end NUMINAMATH_CALUDE_djibo_sister_age_djibo_sister_age_is_28_l1148_114857


namespace NUMINAMATH_CALUDE_trajectory_of_symmetric_point_l1148_114843

/-- The trajectory of point P, symmetric to a point Q on the curve y = x^2 - 2 with respect to point A(1, 0) -/
theorem trajectory_of_symmetric_point :
  let A : ℝ × ℝ := (1, 0)
  let C : ℝ → ℝ := fun x => x^2 - 2
  ∀ Q : ℝ × ℝ, (Q.2 = C Q.1) →
  ∀ P : ℝ × ℝ, (Q.1 = 2 - P.1 ∧ Q.2 = -P.2) →
  P.2 = -P.1^2 + 4*P.1 - 2 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_symmetric_point_l1148_114843


namespace NUMINAMATH_CALUDE_shelf_filling_l1148_114811

theorem shelf_filling (P Q T N K : ℕ) (hP : P > 0) (hQ : Q > 0) (hT : T > 0) (hN : N > 0) (hK : K > 0)
  (hUnique : P ≠ Q ∧ P ≠ T ∧ P ≠ N ∧ P ≠ K ∧ Q ≠ T ∧ Q ≠ N ∧ Q ≠ K ∧ T ≠ N ∧ T ≠ K ∧ N ≠ K)
  (hThicker : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y > x ∧ P * x + Q * y = T * x + N * y ∧ K * x = P * x + Q * y) :
  K = (P * K - T * Q) / (N - Q) :=
by sorry

end NUMINAMATH_CALUDE_shelf_filling_l1148_114811


namespace NUMINAMATH_CALUDE_ratio_to_eleven_l1148_114878

theorem ratio_to_eleven : ∃ x : ℚ, (5 : ℚ) / 1 = x / 11 ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_eleven_l1148_114878


namespace NUMINAMATH_CALUDE_bisecting_chord_equation_l1148_114852

/-- The equation of a line bisecting a chord of a parabola -/
theorem bisecting_chord_equation (x y : ℝ → ℝ) :
  (∀ t, (y t)^2 = 16 * (x t)) →  -- Parabola equation
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ 
    (x t₁ + x t₂) / 2 = 2 ∧ 
    (y t₁ + y t₂) / 2 = 1) →  -- Midpoint condition
  (∃ a b c : ℝ, ∀ t, a * (x t) + b * (y t) + c = 0 ∧ 
    a = 8 ∧ b = -1 ∧ c = -15) := by
sorry

end NUMINAMATH_CALUDE_bisecting_chord_equation_l1148_114852


namespace NUMINAMATH_CALUDE_bucket_pouring_l1148_114819

theorem bucket_pouring (capacity_a capacity_b : ℚ) : 
  capacity_b = (1 / 2) * capacity_a →
  let initial_sand_a := (1 / 4) * capacity_a
  let initial_sand_b := (3 / 8) * capacity_b
  let final_sand_a := initial_sand_a + initial_sand_b
  final_sand_a = (7 / 16) * capacity_a :=
by sorry

end NUMINAMATH_CALUDE_bucket_pouring_l1148_114819


namespace NUMINAMATH_CALUDE_largest_b_value_l1148_114822

theorem largest_b_value (b : ℝ) (h : (3*b + 6)*(b - 2) = 9*b) : b ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_b_value_l1148_114822


namespace NUMINAMATH_CALUDE_existence_of_solution_l1148_114861

theorem existence_of_solution :
  ∃ t : ℝ, Real.exp (1 - 2*t) = 3 * Real.sin (2*t - 2) + Real.cos (2*t) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l1148_114861


namespace NUMINAMATH_CALUDE_smallest_b_value_l1148_114877

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 10) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 25) :
  ∀ b' : ℕ+, b'.val < b.val → 
    ¬(∃ a' : ℕ+, a'.val - b'.val = 10 ∧ 
      Nat.gcd ((a'.val^3 + b'.val^3) / (a'.val + b'.val)) (a'.val * b'.val) = 25) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1148_114877


namespace NUMINAMATH_CALUDE_unique_line_through_sqrt3_and_rationals_l1148_114880

-- Define a point in R²
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through (√3, 0)
structure Line where
  slope : ℝ

def isRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

def linePassesThroughRationalPoints (l : Line) : Prop :=
  ∃ (p1 p2 : Point), p1 ≠ p2 ∧ isRational p1.x ∧ isRational p1.y ∧ 
                     isRational p2.x ∧ isRational p2.y ∧
                     p1.y = l.slope * (p1.x - Real.sqrt 3) ∧
                     p2.y = l.slope * (p2.x - Real.sqrt 3)

theorem unique_line_through_sqrt3_and_rationals :
  ∃! (l : Line), linePassesThroughRationalPoints l :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_sqrt3_and_rationals_l1148_114880


namespace NUMINAMATH_CALUDE_fifth_friend_contribution_l1148_114891

def friend_contribution (total : ℝ) (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = total ∧
  a = (1/2) * (b + c + d + e) ∧
  b = (1/3) * (a + c + d + e) ∧
  c = (1/4) * (a + b + d + e) ∧
  d = (1/5) * (a + b + c + e)

theorem fifth_friend_contribution :
  ∃ a b c d : ℝ, friend_contribution 120 a b c d 52.55 := by
  sorry

end NUMINAMATH_CALUDE_fifth_friend_contribution_l1148_114891


namespace NUMINAMATH_CALUDE_least_lcm_value_l1148_114874

def lcm_problem (a b c : ℕ) : Prop :=
  (Nat.lcm a b = 40) ∧ 
  (Nat.lcm b c = 21) ∧
  (Nat.lcm a c ≥ 24) ∧
  ∀ x y, (Nat.lcm x y = 40) → (Nat.lcm y c = 21) → (Nat.lcm x c ≥ 24)

theorem least_lcm_value : ∃ a b c, lcm_problem a b c ∧ Nat.lcm a c = 24 :=
sorry

end NUMINAMATH_CALUDE_least_lcm_value_l1148_114874


namespace NUMINAMATH_CALUDE_expand_expression_l1148_114884

theorem expand_expression (x : ℝ) : 5 * (-3 * x^3 + 4 * x^2 - 2 * x + 7) = -15 * x^3 + 20 * x^2 - 10 * x + 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1148_114884


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l1148_114873

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  z₁ = 2 + I →
  (z₁.re = -z₂.re ∧ z₁.im = z₂.im) →
  z₁ * z₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l1148_114873


namespace NUMINAMATH_CALUDE_final_position_l1148_114821

/-- Represents the position of the letter F -/
inductive Position
  | PositiveX_PositiveY
  | NegativeX_NegativeY
  | PositiveX_NegativeY
  | NegativeX_PositiveY
  | PositiveXPlusY
  | NegativeXPlusY
  | PositiveXMinusY
  | NegativeXMinusY

/-- Represents the transformations -/
inductive Transformation
  | RotateClockwise (angle : ℝ)
  | ReflectXAxis
  | RotateAroundOrigin (angle : ℝ)

/-- Initial position of F after 90° clockwise rotation -/
def initialPosition : Position := Position.PositiveX_NegativeY

/-- Sequence of transformations -/
def transformations : List Transformation := [
  Transformation.RotateClockwise 45,
  Transformation.ReflectXAxis,
  Transformation.RotateAroundOrigin 180
]

/-- Applies a single transformation to a position -/
def applyTransformation (p : Position) (t : Transformation) : Position :=
  sorry

/-- Applies a sequence of transformations to a position -/
def applyTransformations (p : Position) (ts : List Transformation) : Position :=
  sorry

/-- The final position theorem -/
theorem final_position :
  applyTransformations initialPosition transformations = Position.NegativeXPlusY :=
  sorry

end NUMINAMATH_CALUDE_final_position_l1148_114821


namespace NUMINAMATH_CALUDE_value_of_c_l1148_114870

theorem value_of_c (x b c : ℝ) (h1 : x - 1/x = 2*b) (h2 : x^3 - 1/x^3 = c) : c = 8*b^3 + 6*b := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l1148_114870


namespace NUMINAMATH_CALUDE_problem_solution_l1148_114823

theorem problem_solution : (120 / (6 / 3)) * 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1148_114823


namespace NUMINAMATH_CALUDE_ten_people_two_vip_seats_l1148_114896

/-- The number of ways to arrange n people around a round table with k marked VIP seats,
    where arrangements are considered the same if rotations preserve who sits in the VIP seats -/
def roundTableArrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose k) * (n - k).factorial

/-- Theorem stating that for 10 people and 2 VIP seats, there are 1,814,400 arrangements -/
theorem ten_people_two_vip_seats :
  roundTableArrangements 10 2 = 1814400 := by
  sorry

#eval roundTableArrangements 10 2

end NUMINAMATH_CALUDE_ten_people_two_vip_seats_l1148_114896


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1148_114853

theorem toy_store_revenue_ratio :
  ∀ (N D J : ℝ),
  N > 0 →
  N = (2/5) * D →
  D = 3.75 * ((N + J) / 2) →
  J / N = 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l1148_114853


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1148_114876

/-- Given a geometric sequence {a_n} where a_1 and a_5 are the positive roots of x^2 - 10x + 16 = 0, prove that a_3 = 4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  (a 1 * a 1 - 10 * a 1 + 16 = 0) →  -- a_1 is a root of x^2 - 10x + 16 = 0
  (a 5 * a 5 - 10 * a 5 + 16 = 0) →  -- a_5 is a root of x^2 - 10x + 16 = 0
  (0 < a 1) →  -- a_1 is positive
  (0 < a 5) →  -- a_5 is positive
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1148_114876


namespace NUMINAMATH_CALUDE_not_all_mages_are_wizards_l1148_114813

-- Define the universe of discourse
variable {U : Type}

-- Define predicates for being a mage, sorcerer, and wizard
variable (Mage Sorcerer Wizard : U → Prop)

-- State the theorem
theorem not_all_mages_are_wizards :
  (∃ x, Mage x ∧ ¬Sorcerer x) →
  (∀ x, Mage x ∧ Wizard x → Sorcerer x) →
  ∃ x, Mage x ∧ ¬Wizard x :=
by sorry

end NUMINAMATH_CALUDE_not_all_mages_are_wizards_l1148_114813


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1148_114892

def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂ * x^2 + a₁ * x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  ∀ x : ℤ, polynomial a₂ a₁ x = 0 → x ∈ possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1148_114892


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l1148_114887

theorem value_of_a_minus_b (a b : ℝ) 
  (eq1 : 2010 * a + 2014 * b = 2018)
  (eq2 : 2012 * a + 2016 * b = 2020) : 
  a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l1148_114887


namespace NUMINAMATH_CALUDE_cubic_one_real_solution_l1148_114869

/-- The cubic equation 4x^3 + 9x^2 + kx + 4 = 0 has exactly one real solution if and only if k = 6.75 -/
theorem cubic_one_real_solution (k : ℝ) : 
  (∃! x : ℝ, 4 * x^3 + 9 * x^2 + k * x + 4 = 0) ↔ k = 27/4 := by
sorry

end NUMINAMATH_CALUDE_cubic_one_real_solution_l1148_114869


namespace NUMINAMATH_CALUDE_horner_method_v3_l1148_114854

/-- The polynomial f(x) = 2 + 0.35x + 1.8x² - 3.66x³ + 6x⁴ - 5.2x⁵ + x⁶ -/
def f (x : ℝ) : ℝ := 2 + 0.35*x + 1.8*x^2 - 3.66*x^3 + 6*x^4 - 5.2*x^5 + x^6

/-- Horner's method for calculating v₃ -/
def horner_v3 (x : ℝ) : ℝ :=
  let v0 : ℝ := 1
  let v1 : ℝ := v0 * x - 5.2
  let v2 : ℝ := v1 * x + 6
  v2 * x - 3.66

theorem horner_method_v3 :
  horner_v3 (-1) = -15.86 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1148_114854


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1148_114859

theorem perpendicular_vectors (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![4, -2]
  (∀ i, i < 2 → a i * b i = 0) → m = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1148_114859


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1148_114849

theorem geometric_sequence_first_term 
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r^5 = Nat.factorial 9)  -- 6th term is 9!
  (h2 : a * r^8 = Nat.factorial 10) -- 9th term is 10!
  : a = (Nat.factorial 9) / (10 ^ (5/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1148_114849


namespace NUMINAMATH_CALUDE_mrs_hilt_animal_legs_l1148_114898

/-- The number of legs for each animal type -/
def dog_legs : ℕ := 4
def chicken_legs : ℕ := 2
def spider_legs : ℕ := 8
def octopus_legs : ℕ := 8

/-- The number of each animal type Mrs. Hilt saw -/
def dogs_seen : ℕ := 3
def chickens_seen : ℕ := 4
def spiders_seen : ℕ := 2
def octopuses_seen : ℕ := 1

/-- The total number of animal legs Mrs. Hilt saw -/
def total_legs : ℕ := dogs_seen * dog_legs + chickens_seen * chicken_legs + 
                      spiders_seen * spider_legs + octopuses_seen * octopus_legs

theorem mrs_hilt_animal_legs : total_legs = 44 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_animal_legs_l1148_114898


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_l1148_114824

theorem consecutive_odd_integers (x : ℤ) : 
  (x % 2 = 1) →                           -- x is odd
  ((x + 2) % 2 = 1) →                     -- x + 2 is odd
  ((x + 4) % 2 = 1) →                     -- x + 4 is odd
  ((x + 2) + (x + 4) = x + 17) →          -- sum of last two equals first plus 17
  (x + 4 = 15) :=                         -- third integer is 15
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_l1148_114824


namespace NUMINAMATH_CALUDE_smallest_possible_b_l1148_114875

theorem smallest_possible_b (a b : ℝ) : 
  (1 < a ∧ a < b) →
  (1 + a ≤ b) →
  (1/a + 1/b ≤ 1) →
  b ≥ (3 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l1148_114875


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1148_114833

theorem fraction_to_decimal : (53 : ℚ) / (2^2 * 5^3) = 0.106 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1148_114833


namespace NUMINAMATH_CALUDE_even_sum_squares_half_l1148_114832

theorem even_sum_squares_half (n x y : ℤ) (h : 2 * n = x^2 + y^2) :
  n = ((x + y) / 2)^2 + ((x - y) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_squares_half_l1148_114832


namespace NUMINAMATH_CALUDE_pauls_initial_amount_l1148_114871

/-- The amount of money Paul initially had for shopping --/
def initial_amount : ℕ := 15

/-- The cost of bread --/
def bread_cost : ℕ := 2

/-- The cost of butter --/
def butter_cost : ℕ := 3

/-- The cost of juice (twice the price of bread) --/
def juice_cost : ℕ := 2 * bread_cost

/-- The amount Paul had left after shopping --/
def amount_left : ℕ := 6

/-- Theorem stating that Paul's initial amount equals the sum of his purchases and remaining money --/
theorem pauls_initial_amount :
  initial_amount = bread_cost + butter_cost + juice_cost + amount_left := by
  sorry

end NUMINAMATH_CALUDE_pauls_initial_amount_l1148_114871


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_and_max_distance_and_segment_length_l1148_114839

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := 2*x + (1+m)*y + 2*m = 0

-- Define point P
def P : ℝ × ℝ := (-1, 0)

-- Define point Q
def Q : ℝ × ℝ := (1, -2)

-- Define point N
def N : ℝ × ℝ := (2, 1)

theorem line_passes_through_fixed_point_and_max_distance_and_segment_length :
  (∀ m : ℝ, line_l m Q.1 Q.2) ∧
  (∃ d : ℝ, d = 2 * Real.sqrt 2 ∧ 
    ∀ m : ℝ, ∀ x y : ℝ, line_l m x y → Real.sqrt ((x - P.1)^2 + (y - P.2)^2) ≤ d) ∧
  (∀ M : ℝ × ℝ, (M.1 - 0)^2 + (M.2 + 1)^2 = 2 →
    Real.sqrt 2 ≤ Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) ≤ 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_and_max_distance_and_segment_length_l1148_114839


namespace NUMINAMATH_CALUDE_m_range_l1148_114842

-- Define the point M
def M : ℝ × ℝ := (1, 2)

-- Define the proposition p
def p (m : ℝ) : Prop := M.1 - M.2 + m < 0

-- Define the proposition q
def q (m : ℝ) : Prop := m ≠ -2

-- Define the theorem
theorem m_range (m : ℝ) : p m ∧ q m ↔ m ∈ Set.Ioo (-Real.pi) (-2) ∪ Set.Ioo (-2) 1 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1148_114842


namespace NUMINAMATH_CALUDE_kris_bullying_instances_l1148_114818

/-- The number of days Kris is suspended for each bullying instance -/
def suspension_days_per_instance : ℕ := 3

/-- The total number of fingers and toes a typical person has -/
def typical_person_digits : ℕ := 20

/-- The total number of days Kris has been suspended -/
def total_suspension_days : ℕ := 3 * typical_person_digits

/-- The number of bullying instances Kris is responsible for -/
def bullying_instances : ℕ := total_suspension_days / suspension_days_per_instance

theorem kris_bullying_instances : bullying_instances = 20 := by
  sorry

end NUMINAMATH_CALUDE_kris_bullying_instances_l1148_114818


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1148_114895

theorem min_value_quadratic (x y : ℝ) : x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1148_114895


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1148_114835

theorem cubic_equation_solutions :
  (¬ ∃ (x y : ℕ), x ≠ y ∧ x^3 + 5*y = y^3 + 5*x) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5*y = y^3 + 5*x) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1148_114835


namespace NUMINAMATH_CALUDE_club_members_proof_l1148_114840

theorem club_members_proof (total : ℕ) (left_handed : ℕ) (rock_fans : ℕ) (right_handed_non_rock : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : rock_fans = 18)
  (h4 : right_handed_non_rock = 4)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ x : ℕ, x = 7 ∧ 
    x ≤ left_handed ∧ 
    x ≤ rock_fans ∧
    x + (left_handed - x) + (rock_fans - x) + right_handed_non_rock = total :=
by
  sorry

#check club_members_proof

end NUMINAMATH_CALUDE_club_members_proof_l1148_114840


namespace NUMINAMATH_CALUDE_sugar_mixture_profit_l1148_114890

/-- 
Proves that mixing 41.724 kg of sugar costing Rs. 9 per kg with 21.276 kg of sugar costing Rs. 7 per kg 
results in a 10% gain when selling the mixture at Rs. 9.24 per kg, given that the total weight of the mixture is 63 kg.
-/
theorem sugar_mixture_profit (
  total_weight : ℝ) 
  (sugar_a_cost sugar_b_cost selling_price : ℝ)
  (sugar_a_weight sugar_b_weight : ℝ) :
  total_weight = 63 →
  sugar_a_cost = 9 →
  sugar_b_cost = 7 →
  selling_price = 9.24 →
  sugar_a_weight = 41.724 →
  sugar_b_weight = 21.276 →
  sugar_a_weight + sugar_b_weight = total_weight →
  let total_cost := sugar_a_cost * sugar_a_weight + sugar_b_cost * sugar_b_weight
  let total_revenue := selling_price * total_weight
  total_revenue = 1.1 * total_cost :=
by sorry

end NUMINAMATH_CALUDE_sugar_mixture_profit_l1148_114890


namespace NUMINAMATH_CALUDE_sum_right_angles_rectangle_square_l1148_114881

-- Define a rectangle
def Rectangle := Nat

-- Define a square
def Square := Nat

-- Define the number of right angles in a rectangle
def right_angles_rectangle (r : Rectangle) : Nat := 4

-- Define the number of right angles in a square
def right_angles_square (s : Square) : Nat := 4

-- Theorem: The sum of right angles in a rectangle and a square is 8
theorem sum_right_angles_rectangle_square (r : Rectangle) (s : Square) :
  right_angles_rectangle r + right_angles_square s = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_right_angles_rectangle_square_l1148_114881


namespace NUMINAMATH_CALUDE_abs_difference_sqrt_square_l1148_114805

theorem abs_difference_sqrt_square (x α : ℝ) (h : x < α) :
  |x - Real.sqrt ((x - α)^2)| = α - 2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_sqrt_square_l1148_114805


namespace NUMINAMATH_CALUDE_side_face_area_l1148_114816

/-- A rectangular box with specific properties -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ
  front_face_half_top : width * height = (length * width) / 2
  top_face_one_half_side : length * width = (3 * length * height) / 2
  volume : length * width * height = 5184
  perimeter_ratio : 2 * (length + height) = (12 * (length + width)) / 10

/-- The area of the side face of a box with the given properties is 384 square units -/
theorem side_face_area (b : Box) : b.length * b.height = 384 := by
  sorry

end NUMINAMATH_CALUDE_side_face_area_l1148_114816


namespace NUMINAMATH_CALUDE_point_symmetry_l1148_114889

/-- The line with respect to which we're finding symmetry -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Definition of symmetry with respect to a line -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  symmetry_line midpoint_x midpoint_y ∧
  (y₂ - y₁) / (x₂ - x₁) = -1

theorem point_symmetry :
  symmetric_points (-1) 1 2 (-2) := by sorry

end NUMINAMATH_CALUDE_point_symmetry_l1148_114889


namespace NUMINAMATH_CALUDE_unique_parallel_line_in_plane_l1148_114814

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  parallel_line_plane : Line → Plane → Prop
  line_in_plane : Line → Plane → Prop
  parallel_lines : Line → Line → Prop

/-- The theorem statement -/
theorem unique_parallel_line_in_plane 
  (S : Space3D) (l : S.Line) (α : S.Plane) : 
  (¬ S.parallel_line_plane l α) → 
  (¬ S.line_in_plane l α) → 
  ∃! m : S.Line, S.line_in_plane m α ∧ S.parallel_lines m l :=
sorry

end NUMINAMATH_CALUDE_unique_parallel_line_in_plane_l1148_114814


namespace NUMINAMATH_CALUDE_system_solution_unique_l1148_114808

theorem system_solution_unique : ∃! (x y : ℝ), (3 * x - 4 * y = 12) ∧ (9 * x + 6 * y = -18) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1148_114808


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1148_114851

/-- Given a distance of 200 miles and a speed of 25 miles per hour, the time taken is 8 hours. -/
theorem travel_time_calculation (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 200 ∧ speed = 25 → time = distance / speed → time = 8 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l1148_114851


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l1148_114801

theorem trigonometric_expression_equality : 
  (Real.sin (330 * π / 180) * Real.tan (-13 * π / 3)) / 
  (Real.cos (-19 * π / 6) * Real.cos (690 * π / 180)) = 
  -2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l1148_114801


namespace NUMINAMATH_CALUDE_carson_roller_coaster_rides_l1148_114868

/-- Represents the carnival problem with given wait times and ride frequencies. -/
def carnival_problem (total_time roller_coaster_wait tilt_a_whirl_wait giant_slide_wait : ℕ)
  (tilt_a_whirl_rides giant_slide_rides : ℕ) : Prop :=
  ∃ (roller_coaster_rides : ℕ),
    roller_coaster_rides * roller_coaster_wait +
    tilt_a_whirl_rides * tilt_a_whirl_wait +
    giant_slide_rides * giant_slide_wait = total_time

/-- Theorem stating that Carson rides the roller coaster 4 times. -/
theorem carson_roller_coaster_rides :
  carnival_problem (4 * 60) 30 60 15 1 4 →
  ∃ (roller_coaster_rides : ℕ), roller_coaster_rides = 4 := by
  sorry


end NUMINAMATH_CALUDE_carson_roller_coaster_rides_l1148_114868


namespace NUMINAMATH_CALUDE_triangle_side_length_l1148_114865

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a = 3, C = 120°, and the area of the triangle is 15√3/4, then c = 7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a = 3 → 
  C = 2 * π / 3 → 
  (1/2) * a * b * Real.sin C = (15 * Real.sqrt 3) / 4 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l1148_114865


namespace NUMINAMATH_CALUDE_trash_cans_on_streets_l1148_114826

theorem trash_cans_on_streets (street_cans back_cans : ℕ) : 
  back_cans = 2 * street_cans → 
  street_cans + back_cans = 42 → 
  street_cans = 14 := by
sorry

end NUMINAMATH_CALUDE_trash_cans_on_streets_l1148_114826


namespace NUMINAMATH_CALUDE_work_completion_time_l1148_114837

/-- 
Given that:
- A and B can do the same work
- B can do the work in 16 days
- A and B together can do the work in 16/3 days
Prove that A can do the work alone in 8 days
-/
theorem work_completion_time (b_time a_and_b_time : ℚ) 
  (hb : b_time = 16)
  (hab : a_and_b_time = 16 / 3) : 
  ∃ (a_time : ℚ), a_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1148_114837


namespace NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l1148_114858

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) : 
  a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l1148_114858


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1148_114845

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 2944) (h2 : divisor = 72) (h3 : quotient = 40) :
  dividend - divisor * quotient = 64 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1148_114845


namespace NUMINAMATH_CALUDE_college_student_count_l1148_114879

/-- Given a college with a ratio of boys to girls of 8:5 and 160 girls, 
    the total number of students is 416. -/
theorem college_student_count 
  (ratio_boys : ℕ) 
  (ratio_girls : ℕ) 
  (num_girls : ℕ) 
  (h_ratio : ratio_boys = 8 ∧ ratio_girls = 5)
  (h_girls : num_girls = 160) : 
  (ratio_boys * num_girls / ratio_girls + num_girls : ℕ) = 416 := by
  sorry

#check college_student_count

end NUMINAMATH_CALUDE_college_student_count_l1148_114879


namespace NUMINAMATH_CALUDE_number_problem_l1148_114855

theorem number_problem : ∃ n : ℤ, n - 44 = 15 ∧ n = 59 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1148_114855


namespace NUMINAMATH_CALUDE_investment_rate_problem_l1148_114882

/-- Proves that given the specified conditions, the unknown interest rate is 5% -/
theorem investment_rate_problem (total : ℝ) (first_part : ℝ) (first_rate : ℝ) (total_interest : ℝ)
  (h1 : total = 4000)
  (h2 : first_part = 2800)
  (h3 : first_rate = 3)
  (h4 : total_interest = 144)
  (h5 : first_part * (first_rate / 100) + (total - first_part) * (unknown_rate / 100) = total_interest) :
  unknown_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l1148_114882


namespace NUMINAMATH_CALUDE_tangent_points_sum_constant_l1148_114899

/-- Parabola defined by x^2 = 4y -/
def Parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Point P with coordinates (a, -2) -/
def PointP (a : ℝ) : ℝ × ℝ := (a, -2)

/-- Tangent point on the parabola -/
def TangentPoint (x y : ℝ) : Prop := Parabola x y

/-- The theorem stating that for any point P(a, -2) and two tangent points A(x₁, y₁) and B(x₂, y₂) 
    on the parabola x^2 = 4y, the sum x₁x₂ + y₁y₂ is always equal to -4 -/
theorem tangent_points_sum_constant 
  (a x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : TangentPoint x₁ y₁) 
  (h₂ : TangentPoint x₂ y₂) : 
  x₁ * x₂ + y₁ * y₂ = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_sum_constant_l1148_114899


namespace NUMINAMATH_CALUDE_crayons_division_l1148_114847

/-- Given 24 crayons equally divided among 3 people, prove that each person gets 8 crayons. -/
theorem crayons_division (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  crayons_per_person = total_crayons / num_people →
  crayons_per_person = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayons_division_l1148_114847


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1148_114802

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1148_114802


namespace NUMINAMATH_CALUDE_power_of_two_condition_l1148_114866

theorem power_of_two_condition (n : ℕ+) : 
  (∃ k : ℕ, n.val^3 + n.val - 2 = 2^k) ↔ (n.val = 2 ∨ n.val = 5) := by
sorry

end NUMINAMATH_CALUDE_power_of_two_condition_l1148_114866


namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_45_l1148_114844

theorem ab_plus_cd_equals_45 (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 5)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = 14) :
  a * b + c * d = 45 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_45_l1148_114844


namespace NUMINAMATH_CALUDE_original_item_is_mirror_l1148_114894

-- Define the code language as a function
def code (x : String) : String :=
  match x with
  | "item" => "pencil"
  | "pencil" => "mirror"
  | "mirror" => "board"
  | _ => x

-- Define the useful item to write on paper
def useful_item : String := "pencil"

-- Define the coded useful item
def coded_useful_item : String := "2"

-- Theorem to prove
theorem original_item_is_mirror :
  (code useful_item = coded_useful_item) → 
  (∃ x, code x = useful_item ∧ code (code x) = coded_useful_item) →
  (∃ y, code y = useful_item ∧ y = "mirror") :=
by sorry

end NUMINAMATH_CALUDE_original_item_is_mirror_l1148_114894


namespace NUMINAMATH_CALUDE_max_questions_l1148_114834

/-- Represents a contestant's answers to n questions -/
def Answers (n : ℕ) := Fin n → Bool

/-- The number of contestants -/
def num_contestants : ℕ := 8

/-- Condition: For any pair of questions, exactly two contestants answered each combination -/
def valid_distribution (n : ℕ) (answers : Fin num_contestants → Answers n) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = true ∧ answers k j = true) ∧
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = false ∧ answers k j = false) ∧
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = true ∧ answers k j = false) ∧
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = false ∧ answers k j = true)

/-- The maximum number of questions satisfying the conditions -/
theorem max_questions :
  ∀ n : ℕ, (∃ answers : Fin num_contestants → Answers n, valid_distribution n answers) →
    n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_questions_l1148_114834


namespace NUMINAMATH_CALUDE_f_minus_six_equals_minus_one_l1148_114836

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_minus_six_equals_minus_one 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : is_even f)
  (h2 : has_period f 6)
  (h3 : ∀ x ∈ Set.Icc (-3) 3, f x = (x + 1) * (x - a)) :
  f (-6) = -1 := by
sorry

end NUMINAMATH_CALUDE_f_minus_six_equals_minus_one_l1148_114836


namespace NUMINAMATH_CALUDE_evening_temp_calculation_l1148_114862

/-- Given a noon temperature and a temperature decrease, calculate the evening temperature. -/
def evening_temperature (noon_temp : Int) (decrease : Int) : Int :=
  noon_temp - decrease

/-- Theorem: If the noon temperature is 1℃ and it decreases by 3℃, then the evening temperature is -2℃. -/
theorem evening_temp_calculation :
  evening_temperature 1 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_evening_temp_calculation_l1148_114862


namespace NUMINAMATH_CALUDE_cone_surface_area_minimization_l1148_114810

/-- 
Given a right circular cone with fixed volume V, base radius R, and height H,
prove that H/R = 3 when the total surface area is minimized.
-/
theorem cone_surface_area_minimization (V : ℝ) (V_pos : V > 0) :
  ∃ (R H : ℝ), R > 0 ∧ H > 0 ∧
  (∀ (r h : ℝ), r > 0 → h > 0 → (1/3) * Real.pi * r^2 * h = V →
    R^2 * (Real.pi * R + Real.pi * Real.sqrt (R^2 + H^2)) ≤ 
    r^2 * (Real.pi * r + Real.pi * Real.sqrt (r^2 + h^2))) ∧
  H / R = 3 := by
  sorry


end NUMINAMATH_CALUDE_cone_surface_area_minimization_l1148_114810


namespace NUMINAMATH_CALUDE_intersection_points_range_l1148_114838

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem intersection_points_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) ↔ 
  -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_range_l1148_114838


namespace NUMINAMATH_CALUDE_complex_cube_root_l1148_114846

theorem complex_cube_root (x y : ℕ+) :
  (↑x + ↑y * I : ℂ)^3 = 2 + 11 * I →
  ↑x + ↑y * I = 2 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_root_l1148_114846
