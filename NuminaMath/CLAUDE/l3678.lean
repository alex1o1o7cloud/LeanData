import Mathlib

namespace area_of_triangle_MOI_l3678_367838

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15)
  (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 8)
  (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 17)

/-- Circumcenter of a triangle --/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Center of circle tangent to AC, BC, and circumcircle --/
def tangent_circle_center (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point lies on the internal bisector of angle A --/
def on_angle_bisector (t : Triangle) (p : ℝ × ℝ) : Prop := sorry

/-- Area of a triangle given its vertices --/
def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

/-- Main theorem --/
theorem area_of_triangle_MOI (t : Triangle) :
  let O := circumcenter t
  let I := incenter t
  let M := tangent_circle_center t
  on_angle_bisector t M →
  triangle_area M O I = 4.5 := by sorry

end area_of_triangle_MOI_l3678_367838


namespace julias_preferred_number_l3678_367898

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem julias_preferred_number :
  ∃! n : ℕ,
    100 < n ∧ n < 200 ∧
    is_multiple n 13 ∧
    ¬ is_multiple n 3 ∧
    is_multiple (digit_sum n) 5 ∧
    n = 104 := by
  sorry

end julias_preferred_number_l3678_367898


namespace sequence_inequality_l3678_367857

theorem sequence_inequality (a : ℕ → ℝ) (h1 : ∀ n, a n ≥ 0) 
  (h2 : ∀ m n, a (m + n) ≤ a m + a n) :
  ∀ m n, m ≤ n → a n ≤ m * a 1 + (n / m - 1) * a m :=
by sorry

end sequence_inequality_l3678_367857


namespace iphone_defects_l3678_367889

theorem iphone_defects (
  initial_samsung : ℕ)
  (initial_iphone : ℕ)
  (final_samsung : ℕ)
  (final_iphone : ℕ)
  (total_sold : ℕ)
  (h1 : initial_samsung = 14)
  (h2 : initial_iphone = 8)
  (h3 : final_samsung = 10)
  (h4 : final_iphone = 5)
  (h5 : total_sold = 4)
  : initial_iphone - final_iphone - (total_sold - (initial_samsung - final_samsung)) = 3 :=
by
  sorry

end iphone_defects_l3678_367889


namespace complement_intersection_cardinality_l3678_367878

def U : Finset ℕ := {3,4,5,7,8,9}
def A : Finset ℕ := {4,5,7,8}
def B : Finset ℕ := {3,4,7,8}

theorem complement_intersection_cardinality :
  Finset.card (U \ (A ∩ B)) = 3 := by sorry

end complement_intersection_cardinality_l3678_367878


namespace vector_properties_l3678_367825

/-- Given vectors a, b, c and x ∈ [0,π], prove two statements about x and sin(x + π/6) -/
theorem vector_properties (x : Real) 
  (hx : x ∈ Set.Icc 0 Real.pi)
  (a : Fin 2 → Real)
  (ha : a = fun i => if i = 0 then Real.sin x else Real.sqrt 3 * Real.cos x)
  (b : Fin 2 → Real)
  (hb : b = fun i => if i = 0 then -1 else 1)
  (c : Fin 2 → Real)
  (hc : c = fun i => if i = 0 then 1 else -1) :
  (∃ (k : Real), (a + b) = k • c → x = 5 * Real.pi / 6) ∧
  (a • b = 1 / 2 → Real.sin (x + Real.pi / 6) = Real.sqrt 15 / 4) := by
sorry

end vector_properties_l3678_367825


namespace function_properties_l3678_367895

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem function_properties (α : ℝ) 
  (h1 : 0 < α) (h2 : α < 3 * Real.pi / 4) (h3 : f α = 6 / 5) :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ 
    ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ x : ℝ, f x ≥ -2) ∧
  (∃ x : ℝ, f x = -2) ∧
  f (2 * α) = 31 * Real.sqrt 2 / 25 :=
sorry

end function_properties_l3678_367895


namespace smallest_value_absolute_equation_l3678_367811

theorem smallest_value_absolute_equation :
  (∃ x : ℝ, |x - 8| = 15) ∧
  (∀ x : ℝ, |x - 8| = 15 → x ≥ -7) ∧
  |-7 - 8| = 15 := by
  sorry

end smallest_value_absolute_equation_l3678_367811


namespace water_ratio_corn_to_pig_l3678_367842

def water_pumping_rate : ℚ := 3
def pumping_time : ℕ := 25
def corn_rows : ℕ := 4
def corn_plants_per_row : ℕ := 15
def num_pigs : ℕ := 10
def water_per_pig : ℚ := 4
def num_ducks : ℕ := 20
def water_per_duck : ℚ := 1/4

theorem water_ratio_corn_to_pig :
  let total_water := water_pumping_rate * pumping_time
  let total_corn_plants := corn_rows * corn_plants_per_row
  let water_for_pigs := num_pigs * water_per_pig
  let water_for_ducks := num_ducks * water_per_duck
  let water_for_corn := total_water - water_for_pigs - water_for_ducks
  let water_per_corn := water_for_corn / total_corn_plants
  water_per_corn / water_per_pig = 1/8 := by sorry

end water_ratio_corn_to_pig_l3678_367842


namespace coin_toss_count_l3678_367836

theorem coin_toss_count (total_tosses : ℕ) (tail_count : ℕ) (head_count : ℕ) :
  total_tosses = 14 →
  tail_count = 5 →
  total_tosses = head_count + tail_count →
  head_count = 9 := by
  sorry

end coin_toss_count_l3678_367836


namespace reciprocal_of_negative_nine_l3678_367826

theorem reciprocal_of_negative_nine (x : ℚ) : 
  (x * (-9) = 1) → x = -1/9 := by
  sorry

end reciprocal_of_negative_nine_l3678_367826


namespace gcd_of_315_and_2016_l3678_367882

theorem gcd_of_315_and_2016 : Nat.gcd 315 2016 = 63 := by
  sorry

end gcd_of_315_and_2016_l3678_367882


namespace diophantine_approximation_l3678_367862

theorem diophantine_approximation (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℕ), 1 ≤ b ∧ b ≤ n ∧ |a - b * Real.sqrt 2| ≤ 1 / n := by
  sorry

end diophantine_approximation_l3678_367862


namespace max_residents_per_apartment_is_four_l3678_367817

/-- Represents a block of flats -/
structure BlockOfFlats where
  floors : ℕ
  apartments_per_floor_type1 : ℕ
  apartments_per_floor_type2 : ℕ
  max_residents : ℕ

/-- Calculates the maximum number of residents per apartment -/
def max_residents_per_apartment (block : BlockOfFlats) : ℕ :=
  block.max_residents / ((block.floors / 2) * block.apartments_per_floor_type1 + 
                         (block.floors / 2) * block.apartments_per_floor_type2)

/-- Theorem stating the maximum number of residents per apartment -/
theorem max_residents_per_apartment_is_four (block : BlockOfFlats) 
  (h1 : block.floors = 12)
  (h2 : block.apartments_per_floor_type1 = 6)
  (h3 : block.apartments_per_floor_type2 = 5)
  (h4 : block.max_residents = 264) :
  max_residents_per_apartment block = 4 := by
  sorry

#eval max_residents_per_apartment { 
  floors := 12, 
  apartments_per_floor_type1 := 6, 
  apartments_per_floor_type2 := 5, 
  max_residents := 264 
}

end max_residents_per_apartment_is_four_l3678_367817


namespace complex_power_difference_l3678_367827

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end complex_power_difference_l3678_367827


namespace square_area_to_side_length_ratio_l3678_367835

theorem square_area_to_side_length_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a^2 / b^2 = 72 / 98) → (a / b = 6 / 7) := by
  sorry

end square_area_to_side_length_ratio_l3678_367835


namespace least_positive_integer_divisible_by_53_l3678_367821

theorem least_positive_integer_divisible_by_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧
  (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) ∧ 
  x = 4 := by
  sorry

end least_positive_integer_divisible_by_53_l3678_367821


namespace problem_statement_l3678_367832

theorem problem_statement (a b c d e : ℕ+) 
  (h1 : a * b + a + b = 624)
  (h2 : b * c + b + c = 234)
  (h3 : c * d + c + d = 156)
  (h4 : d * e + d + e = 80)
  (h5 : a * b * c * d * e = 3628800) : -- 3628800 is 10!
  a - e = 22 := by
  sorry

end problem_statement_l3678_367832


namespace morgan_sat_score_l3678_367837

theorem morgan_sat_score (second_score : ℝ) (improvement_rate : ℝ) :
  second_score = 1100 →
  improvement_rate = 0.1 →
  ∃ (first_score : ℝ), first_score * (1 + improvement_rate) = second_score ∧ first_score = 1000 :=
by
  sorry

end morgan_sat_score_l3678_367837


namespace garden_expansion_l3678_367808

/-- Given a rectangular garden with dimensions 50 feet by 20 feet, 
    prove that adding 40 feet of fencing and reshaping into a square 
    results in a garden 1025 square feet larger than the original. -/
theorem garden_expansion (original_length : ℝ) (original_width : ℝ) 
  (additional_fence : ℝ) (h1 : original_length = 50) 
  (h2 : original_width = 20) (h3 : additional_fence = 40) : 
  let original_area := original_length * original_width
  let original_perimeter := 2 * (original_length + original_width)
  let new_perimeter := original_perimeter + additional_fence
  let new_side := new_perimeter / 4
  let new_area := new_side * new_side
  new_area - original_area = 1025 := by
sorry

end garden_expansion_l3678_367808


namespace sum_of_constants_l3678_367830

theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 1 ↔ x = -1)) ∧
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 7 ↔ x = -3)) →
  a + b = 19 := by
sorry

end sum_of_constants_l3678_367830


namespace integral_approximation_l3678_367849

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf_continuous : ContinuousOn f (Set.Icc 0 1))
variable (hf_range : ∀ x ∈ Set.Icc 0 1, 0 ≤ f x ∧ f x ≤ 1)

-- Define N and N_1
variable (N N_1 : ℕ)

-- Define the theorem
theorem integral_approximation :
  ∃ ε > 0, |∫ x in Set.Icc 0 1, f x - (N_1 : ℝ) / N| < ε :=
sorry

end integral_approximation_l3678_367849


namespace valid_paths_count_l3678_367884

/-- Represents a point in the 2D lattice --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a move in the lattice --/
inductive Move
  | Right : Move
  | Up : Move
  | Diagonal : Move
  | LongRight : Move

/-- Checks if a sequence of moves is valid (no right angle turns) --/
def isValidPath (path : List Move) : Bool :=
  sorry

/-- Checks if a path leads from (0,0) to (7,5) --/
def leadsTo7_5 (path : List Move) : Bool :=
  sorry

/-- Counts the number of valid paths from (0,0) to (7,5) --/
def countValidPaths : ℕ :=
  sorry

/-- The main theorem stating that the number of valid paths is N --/
theorem valid_paths_count :
  ∃ N : ℕ, countValidPaths = N :=
sorry

end valid_paths_count_l3678_367884


namespace positive_real_sum_one_inequality_l3678_367829

theorem positive_real_sum_one_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end positive_real_sum_one_inequality_l3678_367829


namespace min_moves_for_identical_contents_l3678_367866

/-- Represents a ball color -/
inductive BallColor
| White
| Black

/-- Represents a box containing balls -/
structure Box where
  white : Nat
  black : Nat

/-- Represents a move: taking a ball from a box and either discarding it or transferring it -/
inductive Move
| Discard : BallColor → Move
| Transfer : BallColor → Move

/-- The initial state of the boxes -/
def initialState : (Box × Box) :=
  ({white := 4, black := 6}, {white := 0, black := 10})

/-- Predicate to check if two boxes have identical contents -/
def identicalContents (box1 box2 : Box) : Prop :=
  box1.white = box2.white ∧ box1.black = box2.black

/-- The minimum number of moves required to guarantee identical contents -/
def minMovesForIdenticalContents : Nat := 15

theorem min_moves_for_identical_contents :
  ∀ (sequence : List Move),
  (∃ (finalState : Box × Box),
    finalState.1.white + finalState.1.black + finalState.2.white + finalState.2.black ≤ 
      initialState.1.white + initialState.1.black + initialState.2.white + initialState.2.black ∧
    identicalContents finalState.1 finalState.2) →
  sequence.length ≥ minMovesForIdenticalContents :=
sorry

end min_moves_for_identical_contents_l3678_367866


namespace upward_parabola_m_value_l3678_367880

/-- If y=(m-1)x^2-2mx+1 is an upward-opening parabola, then m = 2 -/
theorem upward_parabola_m_value (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 - 2 * m * x + 1 = 0 → (m - 1) > 0) → 
  m = 2 := by sorry

end upward_parabola_m_value_l3678_367880


namespace f_minus_g_at_7_l3678_367815

def f : ℝ → ℝ := fun _ ↦ 3

def g : ℝ → ℝ := fun _ ↦ 5

theorem f_minus_g_at_7 : f 7 - g 7 = -2 := by
  sorry

end f_minus_g_at_7_l3678_367815


namespace coefficient_x7y_is_20_l3678_367872

/-- The coefficient of x^7y in the expansion of (x^2 + x + y)^5 -/
def coefficient_x7y (x y : ℕ) : ℕ :=
  (Nat.choose 5 1) * (Nat.choose 4 1) * (Nat.choose 3 3)

/-- Theorem stating that the coefficient of x^7y in (x^2 + x + y)^5 is 20 -/
theorem coefficient_x7y_is_20 :
  ∀ x y, coefficient_x7y x y = 20 := by
  sorry

#eval coefficient_x7y 0 0

end coefficient_x7y_is_20_l3678_367872


namespace complex_product_quadrant_l3678_367893

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (0 < z.re) ∧ (0 < z.im) :=
by
  sorry

end complex_product_quadrant_l3678_367893


namespace no_integer_points_in_sphere_intersection_l3678_367875

theorem no_integer_points_in_sphere_intersection : 
  ¬∃ (x y z : ℤ), (x^2 + y^2 + (z - 10)^2 ≤ 9) ∧ (x^2 + y^2 + (z - 2)^2 ≤ 16) :=
by sorry

end no_integer_points_in_sphere_intersection_l3678_367875


namespace alex_grocery_charge_percentage_l3678_367887

/-- The problem of determining Alex's grocery delivery charge percentage --/
theorem alex_grocery_charge_percentage :
  ∀ (car_cost savings_initial trip_charge trips_made grocery_total charge_percentage : ℚ),
  car_cost = 14600 →
  savings_initial = 14500 →
  trip_charge = (3/2) →
  trips_made = 40 →
  grocery_total = 800 →
  car_cost - savings_initial = trip_charge * trips_made + charge_percentage * grocery_total →
  charge_percentage = (1/20) := by
  sorry

end alex_grocery_charge_percentage_l3678_367887


namespace representation_inequality_l3678_367886

/-- The smallest number of 1s needed to represent a positive integer using only 1s, +, ×, and brackets -/
noncomputable def f (n : ℕ) : ℕ := sorry

/-- The inequality holds for all n > 1 -/
theorem representation_inequality (n : ℕ) (hn : n > 1) :
  3 * Real.log n ≤ Real.log 3 * (f n : ℝ) ∧ Real.log 3 * (f n : ℝ) ≤ 5 * Real.log n := by
  sorry

end representation_inequality_l3678_367886


namespace dans_trip_l3678_367831

/-- The distance from Dan's home to his workplace -/
def distance : ℝ := 160

/-- The time of the usual trip in minutes -/
def usual_time : ℝ := 240

/-- The time spent driving at normal speed on the particular day -/
def normal_speed_time : ℝ := 120

/-- The speed reduction factor due to heavy traffic -/
def speed_reduction : ℝ := 0.75

/-- The total trip time on the particular day -/
def total_time : ℝ := 330

theorem dans_trip :
  distance = distance * (normal_speed_time / usual_time + 
    (total_time - normal_speed_time) / (usual_time / speed_reduction)) := by
  sorry

end dans_trip_l3678_367831


namespace intersection_empty_iff_intersection_equals_A_iff_l3678_367870

-- Define sets A and B
def A (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 2 }
def B : Set ℝ := { x | x ≤ 0 ∨ x ≥ 4 }

-- Theorem 1
theorem intersection_empty_iff (a : ℝ) : A a ∩ B = ∅ ↔ 0 < a ∧ a < 2 := by sorry

-- Theorem 2
theorem intersection_equals_A_iff (a : ℝ) : A a ∩ B = A a ↔ a ≤ -2 ∨ a ≥ 4 := by sorry

end intersection_empty_iff_intersection_equals_A_iff_l3678_367870


namespace polynomial_factorization_l3678_367881

theorem polynomial_factorization (x : ℝ) : 3 * x^2 + 3 * x - 18 = 3 * (x + 3) * (x - 2) := by
  sorry

end polynomial_factorization_l3678_367881


namespace students_not_enrolled_l3678_367850

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 6 := by
  sorry

end students_not_enrolled_l3678_367850


namespace geometric_sequence_ratio_l3678_367885

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : geometric_sequence a q) 
  (h2 : ∀ n, a n > 0) 
  (h3 : q^2 = 4) : 
  (a 3 + a 4) / (a 5 + a 6) = 1/4 := by
sorry

end geometric_sequence_ratio_l3678_367885


namespace limit_point_sequence_a_l3678_367861

def sequence_a (n : ℕ) : ℚ := (n + 1) / n

theorem limit_point_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence_a n - 1| < ε :=
sorry

end limit_point_sequence_a_l3678_367861


namespace rectangle_area_with_inscribed_circle_rectangle_area_is_588_l3678_367806

/-- The area of a rectangle with an inscribed circle of radius 7 and length-to-width ratio of 3:1 -/
theorem rectangle_area_with_inscribed_circle : ℝ :=
  let radius : ℝ := 7
  let length_width_ratio : ℝ := 3
  let diameter : ℝ := 2 * radius
  let width : ℝ := diameter
  let length : ℝ := length_width_ratio * width
  let area : ℝ := length * width
  area

/-- Proof that the area of the rectangle is 588 -/
theorem rectangle_area_is_588 : rectangle_area_with_inscribed_circle = 588 := by
  sorry

end rectangle_area_with_inscribed_circle_rectangle_area_is_588_l3678_367806


namespace profit_share_difference_l3678_367863

/-- Represents the profit share of a party -/
structure ProfitShare where
  numerator : ℕ
  denominator : ℕ
  inv_pos : denominator > 0

/-- Calculates the profit for a given share and total profit -/
def calculate_profit (share : ProfitShare) (total_profit : ℚ) : ℚ :=
  total_profit * (share.numerator : ℚ) / (share.denominator : ℚ)

/-- The problem statement -/
theorem profit_share_difference 
  (total_profit : ℚ)
  (share_x share_y share_z : ProfitShare)
  (h_total : total_profit = 700)
  (h_x : share_x = ⟨1, 3, by norm_num⟩)
  (h_y : share_y = ⟨1, 4, by norm_num⟩)
  (h_z : share_z = ⟨1, 5, by norm_num⟩) :
  let profit_x := calculate_profit share_x total_profit
  let profit_y := calculate_profit share_y total_profit
  let profit_z := calculate_profit share_z total_profit
  let max_profit := max profit_x (max profit_y profit_z)
  let min_profit := min profit_x (min profit_y profit_z)
  ∃ (ε : ℚ), abs (max_profit - min_profit - 7148.93) < ε ∧ ε < 0.01 :=
sorry

end profit_share_difference_l3678_367863


namespace sports_club_overlap_l3678_367851

/-- Given a sports club with the following properties:
  * There are 30 total members
  * 17 members play badminton
  * 21 members play tennis
  * 2 members play neither badminton nor tennis
  This theorem proves that 10 members play both badminton and tennis. -/
theorem sports_club_overlap :
  ∀ (total badminton tennis neither : ℕ),
  total = 30 →
  badminton = 17 →
  tennis = 21 →
  neither = 2 →
  badminton + tennis - total + neither = 10 :=
by sorry

end sports_club_overlap_l3678_367851


namespace complex_fraction_simplification_l3678_367833

theorem complex_fraction_simplification :
  (3 + 4 * Complex.I) / (1 - 2 * Complex.I) = -1 + 2 * Complex.I :=
by sorry

end complex_fraction_simplification_l3678_367833


namespace quadratic_root_sum_product_l3678_367844

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 8 ∧ x * y = 12) →
  p + q = 60 := by
sorry

end quadratic_root_sum_product_l3678_367844


namespace line_parallel_to_AB_through_P_circumcircle_OAB_l3678_367800

-- Define the points
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

-- Theorem for the line equation
theorem line_parallel_to_AB_through_P :
  ∀ x y : ℝ, line_equation x y ↔ 
  (∃ t : ℝ, x = 2 + t * (B.1 - A.1) ∧ y = 3 + t * (B.2 - A.2)) :=
sorry

-- Theorem for the circle equation
theorem circumcircle_OAB :
  ∀ x y : ℝ, circle_equation x y ↔
  (x - O.1)^2 + (y - O.2)^2 = (x - A.1)^2 + (y - A.2)^2 ∧
  (x - O.1)^2 + (y - O.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end line_parallel_to_AB_through_P_circumcircle_OAB_l3678_367800


namespace exponent_multiplication_l3678_367899

theorem exponent_multiplication (a : ℝ) : a^4 * a^3 = a^7 := by sorry

end exponent_multiplication_l3678_367899


namespace perpendicular_parallel_implication_l3678_367841

-- Define a structure for a line in 3D space
structure Line3D where
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_parallel_implication (a b c : Line3D) 
  (h1 : perpendicular a b) (h2 : parallel b c) : perpendicular a c :=
sorry

end perpendicular_parallel_implication_l3678_367841


namespace arithmetic_mean_reciprocals_first_four_primes_l3678_367845

theorem arithmetic_mean_reciprocals_first_four_primes :
  let primes : List ℕ := [2, 3, 5, 7]
  let reciprocals := primes.map (λ x => (1 : ℚ) / x)
  let sum := reciprocals.sum
  let mean := sum / 4
  mean = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l3678_367845


namespace sports_purchase_equation_l3678_367896

/-- Represents the cost of sports equipment purchases -/
structure SportsPurchase where
  volleyball_cost : ℝ  -- Cost of one volleyball in yuan
  shot_put_cost : ℝ    -- Cost of one shot put ball in yuan

/-- Conditions of the sports equipment purchase problem -/
def purchase_conditions (p : SportsPurchase) : Prop :=
  2 * p.volleyball_cost + 3 * p.shot_put_cost = 95 ∧
  5 * p.volleyball_cost + 7 * p.shot_put_cost = 230

/-- The theorem stating that the given system of linear equations 
    correctly represents the sports equipment purchase problem -/
theorem sports_purchase_equation (p : SportsPurchase) :
  purchase_conditions p ↔ 
  (2 * p.volleyball_cost + 3 * p.shot_put_cost = 95 ∧
   5 * p.volleyball_cost + 7 * p.shot_put_cost = 230) :=
by sorry

end sports_purchase_equation_l3678_367896


namespace percentage_problem_l3678_367819

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 800 = (20 / 100) * 650 + 190 → P = 40 := by
  sorry

end percentage_problem_l3678_367819


namespace sum_of_solutions_is_zero_l3678_367876

theorem sum_of_solutions_is_zero (x : ℝ) (h : x^2 - 4 = 36) :
  ∃ y : ℝ, y^2 - 4 = 36 ∧ x + y = 0 :=
by sorry

end sum_of_solutions_is_zero_l3678_367876


namespace john_apple_sales_l3678_367804

/-- Calculates the total money earned from selling apples -/
def apple_sales_revenue 
  (trees_x : ℕ) 
  (trees_y : ℕ) 
  (apples_per_tree : ℕ) 
  (price_per_apple : ℚ) : ℚ :=
  (trees_x * trees_y * apples_per_tree : ℚ) * price_per_apple

/-- Proves that John's apple sales revenue is $30 -/
theorem john_apple_sales : 
  apple_sales_revenue 3 4 5 (1/2) = 30 := by
  sorry

end john_apple_sales_l3678_367804


namespace complex_fraction_simplification_l3678_367814

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i : ℂ) / (1 + i) = (1 : ℂ) / 2 + (i : ℂ) / 2 := by
  sorry

end complex_fraction_simplification_l3678_367814


namespace probability_of_white_ball_l3678_367818

theorem probability_of_white_ball (total_balls : Nat) (red_balls white_balls : Nat) :
  total_balls = red_balls + white_balls + 1 →
  red_balls = 2 →
  white_balls = 3 →
  (white_balls : ℚ) / (total_balls - 1 : ℚ) = 3 / 5 := by
sorry

end probability_of_white_ball_l3678_367818


namespace product_of_five_consecutive_integers_l3678_367858

theorem product_of_five_consecutive_integers (n : ℤ) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = n^5 - n^4 - 5*n^3 + 4*n^2 + 4*n :=
by sorry

end product_of_five_consecutive_integers_l3678_367858


namespace triangle_inequality_l3678_367816

/-- Given a triangle with sides a, b, c and area S, 
    the sum of squares of the sides is greater than or equal to 
    4 times the area multiplied by the square root of 3. 
    Equality holds if and only if the triangle is equilateral. -/
theorem triangle_inequality (a b c S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S > 0)
  (h_S : S = Real.sqrt (s * (s - a) * (s - b) * (s - c))) 
  (h_s : s = (a + b + c) / 2) : 
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 ∧ 
  (a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3 ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_l3678_367816


namespace additional_interest_proof_l3678_367823

/-- Calculate the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time

theorem additional_interest_proof :
  let principal : ℚ := 2500
  let time : ℚ := 2
  let higherRate : ℚ := 18 / 100
  let lowerRate : ℚ := 12 / 100
  simpleInterest principal higherRate time - simpleInterest principal lowerRate time = 300 := by
sorry

end additional_interest_proof_l3678_367823


namespace second_number_calculation_l3678_367890

theorem second_number_calculation (A : ℝ) (X : ℝ) (h1 : A = 1280) 
  (h2 : 0.25 * A = 0.20 * X + 190) : X = 650 := by
  sorry

end second_number_calculation_l3678_367890


namespace different_color_probability_l3678_367846

/-- The probability of drawing two balls of different colors from a box -/
theorem different_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  white_balls = 3 →
  black_balls = 2 →
  (white_balls * black_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2 : ℚ) = 3 / 5 := by
  sorry

end different_color_probability_l3678_367846


namespace min_value_theorem_l3678_367883

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + 3 * b = 6) :
  (2 / a + 3 / b) ≥ 25 / 6 := by
  sorry

end min_value_theorem_l3678_367883


namespace inequality_range_l3678_367892

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ≤ 0 → m * (x^2 - 2*x) * Real.exp x + 1 ≥ Real.exp x) ↔ 
  m ≥ -1/2 := by
sorry

end inequality_range_l3678_367892


namespace unequal_grandchildren_probability_l3678_367874

def num_grandchildren : ℕ := 12

def prob_male : ℚ := 1/2

def prob_female : ℚ := 1/2

theorem unequal_grandchildren_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_outcomes := (num_grandchildren.choose (num_grandchildren / 2))
  (total_outcomes - equal_outcomes) / total_outcomes = 3172/4096 :=
sorry

end unequal_grandchildren_probability_l3678_367874


namespace cube_sum_theorem_l3678_367848

theorem cube_sum_theorem (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := by
sorry

end cube_sum_theorem_l3678_367848


namespace monic_quadratic_root_l3678_367867

theorem monic_quadratic_root (x : ℂ) :
  let p : ℂ → ℂ := λ x => x^2 + 6*x + 12
  p (-3 - Complex.I * Real.sqrt 3) = 0 := by
  sorry

end monic_quadratic_root_l3678_367867


namespace remainder_proof_l3678_367843

theorem remainder_proof (x y r : ℤ) : 
  x > 0 →
  x = 7 * y + r →
  0 ≤ r →
  r < 7 →
  2 * x = 18 * y + 2 →
  11 * y - x = 1 →
  r = 3 := by sorry

end remainder_proof_l3678_367843


namespace square_sum_product_l3678_367865

theorem square_sum_product (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) :
  a^2 + b^2 + a * b = 7 := by
  sorry

end square_sum_product_l3678_367865


namespace function_characterization_l3678_367812

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions
def Condition1 (f : RealFunction) : Prop :=
  ∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v)

def Condition2 (f : RealFunction) : Prop :=
  ∀ u : ℝ, f u ≥ 0

-- State the theorem
theorem function_characterization (f : RealFunction) 
  (h1 : Condition1 f) (h2 : Condition2 f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1/2) :=
sorry

end function_characterization_l3678_367812


namespace factorization_equality_l3678_367871

theorem factorization_equality (a : ℝ) : a^2 - 3*a = a*(a - 3) := by
  sorry

end factorization_equality_l3678_367871


namespace tile_pricing_problem_l3678_367853

/-- Represents the price and discount information for tiles --/
structure TileInfo where
  basePrice : ℝ
  discountRate : ℝ
  discountThreshold : ℕ

/-- Calculates the price for a given quantity of tiles --/
def calculatePrice (info : TileInfo) (quantity : ℕ) : ℝ :=
  if quantity ≥ info.discountThreshold
  then info.basePrice * (1 - info.discountRate) * quantity
  else info.basePrice * quantity

/-- Theorem statement for the tile pricing problem --/
theorem tile_pricing_problem
  (redInfo bluInfo : TileInfo)
  (h1 : calculatePrice redInfo 4000 + calculatePrice bluInfo 6000 = 86000)
  (h2 : calculatePrice redInfo 10000 + calculatePrice bluInfo 3500 = 99000)
  (h3 : redInfo.discountRate = 0.2)
  (h4 : bluInfo.discountRate = 0.1)
  (h5 : redInfo.discountThreshold = 5000)
  (h6 : bluInfo.discountThreshold = 5000) :
  redInfo.basePrice = 8 ∧ bluInfo.basePrice = 10 ∧
  (∃ (redQty bluQty : ℕ),
    redQty + bluQty = 12000 ∧
    bluQty ≥ redQty / 2 ∧
    bluQty ≤ 6000 ∧
    calculatePrice redInfo redQty + calculatePrice bluInfo bluQty = 89800 ∧
    ∀ (r b : ℕ), r + b = 12000 → b ≥ r / 2 → b ≤ 6000 →
      calculatePrice redInfo r + calculatePrice bluInfo b ≥ 89800) :=
sorry

end tile_pricing_problem_l3678_367853


namespace percentage_chain_l3678_367888

theorem percentage_chain (n : ℝ) : 
  (0.20 * 0.15 * 0.40 * 0.30 * 0.50 * n = 180) → n = 1000000 := by
  sorry

end percentage_chain_l3678_367888


namespace y_value_at_x_2_l3678_367877

/-- Given y₁ = x² - 7x + 6, y₂ = 7x - 3, and y = y₁ + xy₂, prove that when x = 2, y = 18. -/
theorem y_value_at_x_2 :
  let y₁ : ℝ → ℝ := λ x => x^2 - 7*x + 6
  let y₂ : ℝ → ℝ := λ x => 7*x - 3
  let y : ℝ → ℝ := λ x => y₁ x + x * y₂ x
  y 2 = 18 := by sorry

end y_value_at_x_2_l3678_367877


namespace max_power_of_two_divides_l3678_367869

theorem max_power_of_two_divides (n : ℕ) (hn : n > 0) :
  (∃ m : ℕ, 3^(2*n+3) + 40*n - 27 = 2^6 * m) ∧
  (∃ n₀ : ℕ, n₀ > 0 ∧ ∀ m : ℕ, 3^(2*n₀+3) + 40*n₀ - 27 ≠ 2^7 * m) :=
sorry

end max_power_of_two_divides_l3678_367869


namespace hyperbola_tangent_angle_bisector_parabola_tangent_angle_bisector_l3678_367864

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  f1 : Point2D  -- First focus
  f2 : Point2D  -- Second focus
  a : ℝ         -- Distance from center to vertex

/-- Represents a parabola -/
structure Parabola where
  f : Point2D    -- Focus
  directrix : Line2D

/-- Returns the angle bisector of three points -/
def angleBisector (p1 p2 p3 : Point2D) : Line2D :=
  sorry

/-- Returns the tangent line to a hyperbola at a given point -/
def hyperbolaTangent (h : Hyperbola) (p : Point2D) : Line2D :=
  sorry

/-- Returns the tangent line to a parabola at a given point -/
def parabolaTangent (p : Parabola) (pt : Point2D) : Line2D :=
  sorry

/-- Theorem: The angle bisector property holds for hyperbola tangents -/
theorem hyperbola_tangent_angle_bisector (h : Hyperbola) (p : Point2D) :
  hyperbolaTangent h p = angleBisector h.f1 p h.f2 :=
sorry

/-- Theorem: The angle bisector property holds for parabola tangents -/
theorem parabola_tangent_angle_bisector (p : Parabola) (pt : Point2D) :
  parabolaTangent p pt = angleBisector p.f pt (Point2D.mk 0 0) :=  -- Assuming (0,0) is on the directrix
sorry

end hyperbola_tangent_angle_bisector_parabola_tangent_angle_bisector_l3678_367864


namespace inequality_system_solutions_l3678_367855

theorem inequality_system_solutions :
  {x : ℕ | 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4} = {0, 1, 2} := by
  sorry

end inequality_system_solutions_l3678_367855


namespace coefficient_x_cubed_in_expansion_l3678_367824

theorem coefficient_x_cubed_in_expansion : 
  (Finset.range 37).sum (fun k => (Nat.choose 36 k) * (1 ^ (36 - k)) * (1 ^ k)) = 7140 := by
  sorry

end coefficient_x_cubed_in_expansion_l3678_367824


namespace right_triangle_segment_ratio_l3678_367834

theorem right_triangle_segment_ratio (a b c r s : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 →
  a^2 + b^2 = c^2 →
  r + s = c →
  a^2 = r * c →
  b^2 = s * c →
  a / b = 2 / 5 →
  r / s = 4 / 25 := by
sorry

end right_triangle_segment_ratio_l3678_367834


namespace smallest_even_triangle_perimeter_l3678_367828

/-- A triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2 * n
  side2 : ℕ := 2 * n + 2
  side3 : ℕ := 2 * n + 4

/-- The triangle inequality for EvenTriangle -/
def satisfiesTriangleInequality (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- The theorem stating the smallest possible perimeter -/
theorem smallest_even_triangle_perimeter :
  ∀ t : EvenTriangle, satisfiesTriangleInequality t →
  ∃ t_min : EvenTriangle, satisfiesTriangleInequality t_min ∧
    perimeter t_min = 18 ∧
    ∀ t' : EvenTriangle, satisfiesTriangleInequality t' →
      perimeter t' ≥ perimeter t_min :=
by
  sorry

end smallest_even_triangle_perimeter_l3678_367828


namespace solve_cubic_equation_l3678_367822

theorem solve_cubic_equation (m : ℝ) : (m - 4)^3 = (1/8)⁻¹ ↔ m = 6 := by
  sorry

end solve_cubic_equation_l3678_367822


namespace jerrys_action_figures_l3678_367856

theorem jerrys_action_figures (initial : ℕ) : 
  initial + 11 - 10 = 8 → initial = 7 := by
  sorry

end jerrys_action_figures_l3678_367856


namespace greatest_root_of_g_l3678_367859

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 20 * x^4 - 18 * x^2 + 3

-- State the theorem
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 15 / 5 ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
by sorry

end greatest_root_of_g_l3678_367859


namespace problem_statement_l3678_367803

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 - y^2 = 3*x*y) :
  x^2 / y^2 + y^2 / x^2 - 2 = 9 := by
  sorry

end problem_statement_l3678_367803


namespace rectangle_area_l3678_367840

/-- The area of a rectangle composed of 24 congruent squares arranged in a 6x4 grid, with a diagonal of 10 cm, is 600/13 square cm. -/
theorem rectangle_area (diagonal : ℝ) (rows columns : ℕ) (num_squares : ℕ) : 
  diagonal = 10 → 
  rows = 6 → 
  columns = 4 → 
  num_squares = 24 → 
  (rows * columns : ℝ) * (diagonal^2 / ((rows : ℝ)^2 + (columns : ℝ)^2)) = 600 / 13 := by
  sorry

end rectangle_area_l3678_367840


namespace total_height_increase_l3678_367809

-- Define the increase in height per decade
def height_increase_per_decade : ℝ := 75

-- Define the number of centuries
def num_centuries : ℕ := 4

-- Define the number of decades in a century
def decades_per_century : ℕ := 10

-- Theorem statement
theorem total_height_increase : 
  height_increase_per_decade * (num_centuries * decades_per_century) = 3000 :=
by sorry

end total_height_increase_l3678_367809


namespace square_perimeter_proof_l3678_367852

theorem square_perimeter_proof (p1 p2 p3 : ℝ) : 
  p1 = 60 ∧ p2 = 48 ∧ p3 = 36 →
  (p1 / 4)^2 - (p2 / 4)^2 = (p3 / 4)^2 →
  p3 = 36 := by
sorry

end square_perimeter_proof_l3678_367852


namespace root_exists_in_interval_l3678_367810

def f (x : ℝ) : ℝ := x^5 - x - 1

theorem root_exists_in_interval :
  ∃ r ∈ Set.Ioo 1 2, f r = 0 :=
by sorry

end root_exists_in_interval_l3678_367810


namespace solution_product_l3678_367868

/-- Given that p and q are the two distinct solutions of the equation
    (x - 5)(3x + 9) = x^2 - 16x + 55, prove that (p + 4)(q + 4) = -54 -/
theorem solution_product (p q : ℝ) : 
  (p - 5) * (3 * p + 9) = p^2 - 16 * p + 55 →
  (q - 5) * (3 * q + 9) = q^2 - 16 * q + 55 →
  p ≠ q →
  (p + 4) * (q + 4) = -54 := by
sorry

end solution_product_l3678_367868


namespace system_one_solution_system_two_solution_l3678_367847

-- System 1
theorem system_one_solution (x y : ℚ) : 
  x + y = 4 ∧ 5 * (x - y) - 2 * (x + y) = -1 → x = 27/10 ∧ y = 13/10 := by sorry

-- System 2
theorem system_two_solution (x y : ℚ) :
  2 * (x - y) / 3 - (x + y) / 4 = -1/12 ∧ 3 * (x + y) - 2 * (2 * x - y) = 3 → x = 2 ∧ y = 1 := by sorry

end system_one_solution_system_two_solution_l3678_367847


namespace hotel_charge_difference_l3678_367894

theorem hotel_charge_difference (G R P : ℝ) 
  (hR : R = G * (1 + 0.125))
  (hP : P = R * (1 - 0.2)) :
  P = G * 0.9 := by
sorry

end hotel_charge_difference_l3678_367894


namespace valve_flow_rate_difference_l3678_367839

/-- The problem of calculating the difference in water flow rates between two valves filling a pool. -/
theorem valve_flow_rate_difference (pool_capacity : ℝ) (both_valves_time : ℝ) (first_valve_time : ℝ) :
  pool_capacity = 12000 ∧ 
  both_valves_time = 48 ∧ 
  first_valve_time = 120 →
  (pool_capacity / both_valves_time) - (pool_capacity / first_valve_time) = 50 := by
sorry

end valve_flow_rate_difference_l3678_367839


namespace divisibility_and_smallest_m_l3678_367820

def E (x y m : ℕ) : ℤ := (72 / x)^m + (72 / y)^m - x^m - y^m

theorem divisibility_and_smallest_m :
  ∀ k : ℕ,
  let m := 400 * k + 200
  2005 ∣ E 3 12 m ∧
  2005 ∣ E 9 6 m ∧
  (∀ m' : ℕ, m' > 0 ∧ m' < 200 → ¬(2005 ∣ E 3 12 m' ∧ 2005 ∣ E 9 6 m')) :=
by sorry

end divisibility_and_smallest_m_l3678_367820


namespace geometric_progression_condition_l3678_367860

def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_progression_condition
  (a : ℕ → ℝ)
  (h : ∀ n : ℕ, a (n + 2) = a n * a (n + 1) - c)
  (c : ℝ) :
  is_geometric_progression a ↔ (a 1 = a 2 ∧ c = 0) :=
sorry

end geometric_progression_condition_l3678_367860


namespace line_through_intersection_and_parallel_l3678_367897

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0
def line2 (x y : ℝ) : Prop := 7*x + 15*y + 1 = 0
def line3 (x y : ℝ) : Prop := x + 2*y - 3 = 0
def result_line (x y : ℝ) : Prop := 3*x + 6*y - 2 = 0

-- Theorem statement
theorem line_through_intersection_and_parallel :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧  -- Intersection point satisfies both line1 and line2
    (∀ (x y : ℝ), line3 x y ↔ ∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ y - y₀ = -1/2 * (x - x₀)) ∧  -- line3 has slope -1/2
    (∀ (x y : ℝ), result_line x y ↔ ∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ y - y₀ = -1/2 * (x - x₀)) ∧  -- result_line has slope -1/2
    result_line x₀ y₀  -- result_line passes through the intersection point
  := by sorry

end line_through_intersection_and_parallel_l3678_367897


namespace gcd_lcm_identity_l3678_367813

theorem gcd_lcm_identity (a b c : ℕ+) :
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) =
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) :=
by sorry

end gcd_lcm_identity_l3678_367813


namespace bowling_record_proof_l3678_367805

/-- The old record average score per player per round in a bowling league -/
def old_record : ℝ := 287

/-- Number of players in a team -/
def players_per_team : ℕ := 4

/-- Number of rounds in a season -/
def rounds_per_season : ℕ := 10

/-- Total score of the team after 9 rounds -/
def score_after_nine_rounds : ℕ := 10440

/-- Difference between old record and minimum average needed in final round -/
def score_difference : ℕ := 27

theorem bowling_record_proof :
  old_record = 
    (score_after_nine_rounds + players_per_team * (old_record - score_difference)) / 
    (players_per_team * rounds_per_season) := by
  sorry

end bowling_record_proof_l3678_367805


namespace expression_simplification_l3678_367802

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (a - 1) / (a^2 - 2*a + 1) / ((a^2 + a) / (a^2 - 1) + 1 / (a - 1)) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l3678_367802


namespace dot_product_equals_one_l3678_367873

/-- Given two vectors a and b in ℝ², prove that their dot product is 1. -/
theorem dot_product_equals_one (a b : ℝ × ℝ) : 
  a = (2, 1) → a - 2 • b = (1, 1) → a.fst * b.fst + a.snd * b.snd = 1 := by sorry

end dot_product_equals_one_l3678_367873


namespace chenny_spoons_count_l3678_367879

/-- Proves that Chenny bought 4 spoons given the conditions of the problem -/
theorem chenny_spoons_count : 
  ∀ (num_plates : ℕ) (plate_cost spoon_cost total_cost : ℚ),
    num_plates = 9 →
    plate_cost = 2 →
    spoon_cost = 3/2 →
    total_cost = 24 →
    (total_cost - (↑num_plates * plate_cost)) / spoon_cost = 4 :=
by sorry

end chenny_spoons_count_l3678_367879


namespace calories_per_shake_johns_shake_calories_l3678_367891

/-- Calculates the calories in each shake given John's daily meal plan. -/
theorem calories_per_shake (breakfast : ℕ) (total_daily : ℕ) : ℕ :=
  let lunch := breakfast + breakfast / 4
  let dinner := 2 * lunch
  let meals_total := breakfast + lunch + dinner
  let shakes_total := total_daily - meals_total
  shakes_total / 3

/-- Proves that each shake contains 300 calories given John's meal plan. -/
theorem johns_shake_calories :
  calories_per_shake 500 3275 = 300 := by
  sorry

end calories_per_shake_johns_shake_calories_l3678_367891


namespace quadratic_solution_l3678_367801

/-- A quadratic function passing through specific points with given conditions -/
def QuadraticProblem (f : ℝ → ℝ) : Prop :=
  (∃ b c : ℝ, ∀ x, f x = x^2 + b*x + c) ∧ 
  f 0 = -1 ∧
  f 2 = 7 ∧
  ∃ y₁ y₂ : ℝ, f (-5) = y₁ ∧ ∃ m : ℝ, f m = y₂ ∧ y₁ + y₂ = 28

/-- The solution to the quadratic problem -/
theorem quadratic_solution (f : ℝ → ℝ) (h : QuadraticProblem f) :
  (∀ x, f x = x^2 + 2*x - 1) ∧ 
  (- (2 / (2 * 1)) = -1) ∧
  (∃ m : ℝ, f m = 14 ∧ m = 3) :=
sorry

end quadratic_solution_l3678_367801


namespace unique_number_property_l3678_367807

theorem unique_number_property : ∃! x : ℚ, x / 3 = x - 5 := by sorry

end unique_number_property_l3678_367807


namespace inequality_solution_set_l3678_367854

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) * (x + 3) > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ x > 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set := by sorry

end inequality_solution_set_l3678_367854
