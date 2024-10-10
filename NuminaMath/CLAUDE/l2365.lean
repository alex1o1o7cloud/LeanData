import Mathlib

namespace maurice_cookout_beef_per_package_l2365_236532

/-- Calculates the amount of ground beef per package for Maurice's cookout -/
theorem maurice_cookout_beef_per_package 
  (total_people : ℕ) 
  (beef_per_person : ℕ) 
  (num_packages : ℕ) 
  (h1 : total_people = 10) 
  (h2 : beef_per_person = 2) 
  (h3 : num_packages = 4) : 
  (total_people * beef_per_person) / num_packages = 5 := by
  sorry

#check maurice_cookout_beef_per_package

end maurice_cookout_beef_per_package_l2365_236532


namespace type_b_machine_time_l2365_236544

def job_completion_time (machine_q : ℝ) (machine_b : ℝ) (combined_time : ℝ) : Prop :=
  2 / machine_q + 3 / machine_b = 1 / combined_time

theorem type_b_machine_time : 
  ∀ (machine_b : ℝ),
    job_completion_time 5 machine_b 1.2 →
    machine_b = 90 / 13 := by
  sorry

end type_b_machine_time_l2365_236544


namespace calculate_S_l2365_236513

-- Define the relationship between R, S, and T
def relation (c : ℝ) (R S T : ℝ) : Prop :=
  R = c * (S^2 / T^2)

-- Define the theorem
theorem calculate_S (c : ℝ) (R₁ S₁ T₁ R₂ T₂ : ℝ) :
  relation c R₁ S₁ T₁ →
  R₁ = 9 →
  S₁ = 2 →
  T₁ = 3 →
  R₂ = 16 →
  T₂ = 4 →
  ∃ S₂, relation c R₂ S₂ T₂ ∧ S₂ = 32/9 := by
  sorry

end calculate_S_l2365_236513


namespace average_minutes_run_is_16_l2365_236518

/-- Represents the average number of minutes run per day for each grade --/
structure GradeRunningAverage where
  sixth : ℝ
  seventh : ℝ
  eighth : ℝ

/-- Represents the ratio of students in each grade --/
structure GradeRatio where
  sixth_to_eighth : ℝ
  sixth_to_seventh : ℝ

/-- Calculates the average number of minutes run per day by all students --/
def average_minutes_run (avg : GradeRunningAverage) (ratio : GradeRatio) : ℝ :=
  sorry

/-- Theorem stating that the average number of minutes run per day is 16 --/
theorem average_minutes_run_is_16 (avg : GradeRunningAverage) (ratio : GradeRatio) 
  (h1 : avg.sixth = 16)
  (h2 : avg.seventh = 18)
  (h3 : avg.eighth = 12)
  (h4 : ratio.sixth_to_eighth = 3)
  (h5 : ratio.sixth_to_seventh = 1.5) :
  average_minutes_run avg ratio = 16 := by
  sorry

end average_minutes_run_is_16_l2365_236518


namespace min_zeros_odd_periodic_function_l2365_236574

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem min_zeros_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : has_period f (2 * Real.pi)) 
  (h_zero_3 : f 3 = 0) 
  (h_zero_4 : f 4 = 0) : 
  ∃ (zeros : Finset ℝ), 
    (∀ x ∈ zeros, x ∈ Set.Icc 0 10 ∧ f x = 0) ∧ 
    Finset.card zeros ≥ 11 := by
  sorry

end min_zeros_odd_periodic_function_l2365_236574


namespace three_folds_halved_cut_segments_l2365_236502

/-- A rope folded into equal parts, then folded in half, and cut in the middle -/
structure FoldedRope where
  initial_folds : ℕ  -- number of initial equal folds
  halved : Bool      -- whether the rope is folded in half after initial folding
  cut : Bool         -- whether the rope is cut in the middle

/-- Calculate the number of segments after folding and cutting -/
def num_segments (rope : FoldedRope) : ℕ :=
  if rope.halved ∧ rope.cut then
    rope.initial_folds * 2 + 1
  else
    rope.initial_folds

/-- Theorem: A rope folded into 3 equal parts, then folded in half, and cut in the middle results in 7 segments -/
theorem three_folds_halved_cut_segments :
  ∀ (rope : FoldedRope), rope.initial_folds = 3 → rope.halved → rope.cut →
  num_segments rope = 7 :=
by
  sorry


end three_folds_halved_cut_segments_l2365_236502


namespace total_readers_l2365_236582

/-- The number of eBook readers Anna bought -/
def anna_readers : ℕ := 50

/-- The difference between Anna's and John's initial number of eBook readers -/
def reader_difference : ℕ := 15

/-- The number of eBook readers John lost -/
def john_lost : ℕ := 3

/-- Theorem: The total number of eBook readers John and Anna have is 82 -/
theorem total_readers : 
  anna_readers + (anna_readers - reader_difference - john_lost) = 82 := by
sorry

end total_readers_l2365_236582


namespace power_plus_sum_l2365_236581

theorem power_plus_sum : 10^2 + 10 + 1 = 111 := by
  sorry

end power_plus_sum_l2365_236581


namespace square_side_length_l2365_236549

-- Define the right triangle PQR
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the square on the hypotenuse
structure SquareOnHypotenuse where
  triangle : RightTriangle
  side_length : ℝ
  on_hypotenuse : side_length ≤ triangle.hypotenuse
  vertex_on_legs : ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ triangle.leg1 ∧ 0 ≤ y ∧ y ≤ triangle.leg2 ∧
    x^2 + y^2 = side_length^2

-- Theorem statement
theorem square_side_length (t : RightTriangle) (s : SquareOnHypotenuse) 
  (h1 : t.leg1 = 5) (h2 : t.leg2 = 12) (h3 : s.triangle = t) :
  s.side_length = 480.525 / 101.925 := by
  sorry

end square_side_length_l2365_236549


namespace trip_cost_difference_l2365_236536

def trip_cost_sharing (alice_paid bob_paid charlie_paid dex_paid : ℚ) : ℚ :=
  let total_paid := alice_paid + bob_paid + charlie_paid + dex_paid
  let fair_share := total_paid / 4
  let alice_owes := max (fair_share - alice_paid) 0
  let charlie_owes := max (fair_share - charlie_paid) 0
  let bob_receives := max (bob_paid - fair_share) 0
  min alice_owes bob_receives - min charlie_owes (bob_receives - min alice_owes bob_receives)

theorem trip_cost_difference :
  trip_cost_sharing 160 220 190 95 = -35/2 :=
by sorry

end trip_cost_difference_l2365_236536


namespace quadratic_intersects_x_axis_l2365_236556

/-- A quadratic function of the form y = kx^2 - 7x - 7 -/
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 7 * x - 7

/-- The discriminant of the quadratic function -/
def discriminant (k : ℝ) : ℝ := 49 + 28 * k

/-- Theorem stating the conditions for the quadratic function to intersect the x-axis -/
theorem quadratic_intersects_x_axis (k : ℝ) :
  (∃ x, quadratic_function k x = 0) ↔ (k ≥ -7/4 ∧ k ≠ 0) :=
sorry

end quadratic_intersects_x_axis_l2365_236556


namespace gretchens_earnings_l2365_236551

/-- The amount Gretchen charges per drawing -/
def price_per_drawing : ℕ := 20

/-- The number of drawings sold on Saturday -/
def saturday_sales : ℕ := 24

/-- The number of drawings sold on Sunday -/
def sunday_sales : ℕ := 16

/-- Gretchen's total earnings over the weekend -/
def total_earnings : ℕ := price_per_drawing * (saturday_sales + sunday_sales)

/-- Theorem stating that Gretchen's total earnings are $800 -/
theorem gretchens_earnings : total_earnings = 800 := by
  sorry

end gretchens_earnings_l2365_236551


namespace amp_2_neg1_4_l2365_236511

-- Define the operation &
def amp (a b c : ℝ) : ℝ := b^3 - 3*a*b*c - 4*a*c^2

-- Theorem statement
theorem amp_2_neg1_4 : amp 2 (-1) 4 = -105 := by
  sorry

end amp_2_neg1_4_l2365_236511


namespace f_upper_bound_implies_a_bound_l2365_236550

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - x) * Real.exp x + a * (x - 1)^2

theorem f_upper_bound_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 2 * Real.exp x) →
  a ≤ ((1 - Real.sqrt 2) * Real.exp (1 - Real.sqrt 2)) / 2 :=
by sorry

end f_upper_bound_implies_a_bound_l2365_236550


namespace triangle_centroid_theorem_l2365_236531

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Point O inside triangle ABC -/
structure PointInTriangle (t : Triangle) where
  O : Vector2D
  A : Vector2D
  B : Vector2D
  C : Vector2D

theorem triangle_centroid_theorem (t : Triangle) (p : PointInTriangle t) 
  (h1 : t.b = 6)
  (h2 : t.a * t.c * Real.cos t.B = t.a^2 - t.b^2 + (Real.sqrt 7 / 4) * t.b * t.c)
  (h3 : p.O.x + p.A.x + p.B.x + p.C.x = 0 ∧ p.O.y + p.A.y + p.B.y + p.C.y = 0)
  (h4 : Real.cos (t.A - π/6) = Real.cos t.A * Real.cos (π/6) + Real.sin t.A * Real.sin (π/6)) :
  (p.O.x - p.A.x)^2 + (p.O.y - p.A.y)^2 = 9 := by
  sorry

end triangle_centroid_theorem_l2365_236531


namespace equation_solutions_l2365_236546

theorem equation_solutions : 
  let f (x : ℝ) := 1 / (x^2 + 14*x - 10) + 1 / (x^2 + 3*x - 10) + 1 / (x^2 - 16*x - 10)
  {x : ℝ | f x = 0} = {5, -2, 2, -5} := by
  sorry

end equation_solutions_l2365_236546


namespace f_neg_ten_eq_two_l2365_236579

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 12

-- State the theorem
theorem f_neg_ten_eq_two : f (-10) = 2 := by
  sorry

end f_neg_ten_eq_two_l2365_236579


namespace benny_piggy_bank_l2365_236567

theorem benny_piggy_bank (january_savings : ℕ) (february_savings : ℕ) (total_savings : ℕ) : 
  january_savings = 19 →
  february_savings = 19 →
  total_savings = 46 →
  total_savings - (january_savings + february_savings) = 8 := by
sorry

end benny_piggy_bank_l2365_236567


namespace polynomial_coefficient_product_l2365_236585

theorem polynomial_coefficient_product (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a * (a₁ + a₃) = 40 := by
  sorry

end polynomial_coefficient_product_l2365_236585


namespace parallelogram_area_l2365_236566

def v : Fin 2 → ℝ := ![3, -7]
def w : Fin 2 → ℝ := ![6, 4]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; w 0, w 1]) = 54 := by sorry

end parallelogram_area_l2365_236566


namespace simplify_complex_fraction_l2365_236595

theorem simplify_complex_fraction (a b : ℝ) 
  (h1 : a ≠ b) (h2 : a ≠ -b) (h3 : a ≠ 2*b) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
sorry

end simplify_complex_fraction_l2365_236595


namespace triangle_inequality_l2365_236564

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (Real.sqrt (b + c - a) / (Real.sqrt b + Real.sqrt c - Real.sqrt a)) +
  (Real.sqrt (c + a - b) / (Real.sqrt c + Real.sqrt a - Real.sqrt b)) +
  (Real.sqrt (a + b - c) / (Real.sqrt a + Real.sqrt b - Real.sqrt c)) ≤ 3 := by
sorry

end triangle_inequality_l2365_236564


namespace infinite_solutions_for_primes_l2365_236516

theorem infinite_solutions_for_primes (p : ℕ) (hp : Prime p) :
  Set.Infinite {n : ℕ | n > 0 ∧ p ∣ 2^n - n} :=
sorry

end infinite_solutions_for_primes_l2365_236516


namespace sqrt_sum_fractions_simplification_l2365_236577

theorem sqrt_sum_fractions_simplification :
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_sum_fractions_simplification_l2365_236577


namespace intersection_M_complement_N_l2365_236504

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (U \ N) = {x : ℝ | x < -1} := by sorry

end intersection_M_complement_N_l2365_236504


namespace ellipse_k_range_l2365_236571

/-- Represents an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse (k : ℝ) where
  (equation : ∀ x y : ℝ, x^2 + k*y^2 = 2)
  (is_ellipse : k ≠ 0)
  (foci_on_y : k < 1)

/-- The range of k for an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
theorem ellipse_k_range (k : ℝ) (e : Ellipse k) : 0 < k ∧ k < 1 := by
  sorry

end ellipse_k_range_l2365_236571


namespace stone_counting_l2365_236599

/-- Represents the number of stones in the sequence -/
def num_stones : ℕ := 11

/-- Represents the length of a complete counting cycle -/
def cycle_length : ℕ := 2 * num_stones - 2

/-- The number we want to find the corresponding stone for -/
def target_number : ℕ := 123

/-- The initial number of the stone that corresponds to the target number -/
def corresponding_stone : ℕ := 3

theorem stone_counting (n : ℕ) :
  n % cycle_length = corresponding_stone - 1 →
  ∃ (k : ℕ), n = k * cycle_length + corresponding_stone :=
sorry

end stone_counting_l2365_236599


namespace problem_solution_l2365_236554

theorem problem_solution : (3^5 + 9720) * (Real.sqrt 289 - (845 / 169.1)) = 119556 := by
  sorry

end problem_solution_l2365_236554


namespace bo_number_l2365_236538

theorem bo_number (a b : ℂ) : 
  a * b = 52 - 28 * I ∧ a = 7 + 4 * I → b = 476 / 65 - 404 / 65 * I :=
by sorry

end bo_number_l2365_236538


namespace vector_perpendicular_to_difference_l2365_236542

/-- Given vectors a = (-1, 2) and b = (1, 3), prove that a is perpendicular to (a - b) -/
theorem vector_perpendicular_to_difference (a b : ℝ × ℝ) :
  a = (-1, 2) →
  b = (1, 3) →
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) :=
by sorry

end vector_perpendicular_to_difference_l2365_236542


namespace unique_polynomial_mapping_l2365_236521

/-- A second-degree polynomial in two variables -/
def p (x y : ℕ) : ℕ := ((x + y)^2 + 3*x + y) / 2

/-- Theorem stating the existence of a unique mapping for non-negative integers -/
theorem unique_polynomial_mapping :
  ∀ n : ℕ, ∃! (k m : ℕ), p k m = n :=
sorry

end unique_polynomial_mapping_l2365_236521


namespace trigonometric_identities_l2365_236514

theorem trigonometric_identities :
  (∃ (tan10 tan20 tan23 tan37 : ℝ),
    tan10 = Real.tan (10 * π / 180) ∧
    tan20 = Real.tan (20 * π / 180) ∧
    tan23 = Real.tan (23 * π / 180) ∧
    tan37 = Real.tan (37 * π / 180) ∧
    tan10 * tan20 + Real.sqrt 3 * (tan10 + tan20) = 1 ∧
    tan23 + tan37 + Real.sqrt 3 * tan23 * tan37 = Real.sqrt 3) :=
by sorry

end trigonometric_identities_l2365_236514


namespace age_ratio_ten_years_ago_l2365_236508

-- Define Alice's current age
def alice_current_age : ℕ := 30

-- Define the age difference between Alice and Tom
def age_difference : ℕ := 15

-- Define the number of years that have passed
def years_passed : ℕ := 10

-- Define Tom's current age
def tom_current_age : ℕ := alice_current_age - age_difference

-- Define Alice's age 10 years ago
def alice_past_age : ℕ := alice_current_age - years_passed

-- Define Tom's age 10 years ago
def tom_past_age : ℕ := tom_current_age - years_passed

-- Theorem to prove
theorem age_ratio_ten_years_ago :
  alice_past_age / tom_past_age = 4 :=
sorry

end age_ratio_ten_years_ago_l2365_236508


namespace frustum_volume_ratio_l2365_236575

/-- Given a right prism with square base of side length L₁ and height H, 
    and a frustum of a pyramid extracted from it with square bases of 
    side lengths L₁ (lower) and L₂ (upper) and height H, 
    if the volume of the frustum is 2/3 of the total volume of the prism, 
    then L₁/L₂ = (1 + √5) / 2 -/
theorem frustum_volume_ratio (L₁ L₂ H : ℝ) (h_positive : L₁ > 0 ∧ L₂ > 0 ∧ H > 0) :
  (H / 3 * (L₁^2 + L₁*L₂ + L₂^2)) = (2 / 3 * L₁^2 * H) → 
  L₁ / L₂ = (1 + Real.sqrt 5) / 2 := by
  sorry

end frustum_volume_ratio_l2365_236575


namespace no_consecutive_fourth_powers_l2365_236537

theorem no_consecutive_fourth_powers (n : ℤ) : 
  n^4 + (n+1)^4 + (n+2)^4 + (n+3)^4 ≠ (n+4)^4 := by
  sorry

end no_consecutive_fourth_powers_l2365_236537


namespace irreducible_fraction_l2365_236507

theorem irreducible_fraction (n : ℕ) :
  (Nat.gcd (n^3 + n) (2*n + 1) = 1) ↔ (n % 5 ≠ 2) := by
  sorry

end irreducible_fraction_l2365_236507


namespace costume_cost_is_660_l2365_236568

/-- Represents the cost of materials for Jenna's costume --/
def costume_cost : ℝ :=
  let velvet_price := 3
  let silk_price := 6
  let lace_price := 10
  let satin_price := 4
  let leather_price := 5
  let wool_price := 8
  let ribbon_price := 2

  let skirt_area := 12 * 4
  let skirts_count := 3
  let bodice_silk_area := 2
  let bodice_lace_area := 5 * 2
  let bonnet_area := 2.5 * 1.5
  let shoe_cover_area := 1 * 1.5 * 2
  let cape_area := 5 * 2
  let ribbon_length := 3

  let velvet_cost := velvet_price * skirt_area * skirts_count
  let bodice_cost := silk_price * bodice_silk_area + lace_price * bodice_lace_area
  let bonnet_cost := satin_price * bonnet_area
  let shoe_covers_cost := leather_price * shoe_cover_area
  let cape_cost := wool_price * cape_area
  let ribbon_cost := ribbon_price * ribbon_length

  velvet_cost + bodice_cost + bonnet_cost + shoe_covers_cost + cape_cost + ribbon_cost

/-- Theorem stating that the total cost of Jenna's costume materials is $660 --/
theorem costume_cost_is_660 : costume_cost = 660 := by
  sorry

end costume_cost_is_660_l2365_236568


namespace range_of_a_l2365_236515

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |2 - x| + |1 + x| ≥ a^2 - 2*a) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end range_of_a_l2365_236515


namespace largest_value_l2365_236580

theorem largest_value (a b c d e : ℝ) 
  (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5 ∧ a - 2 = e - 6) :
  e = max a (max b (max c d)) :=
by sorry

end largest_value_l2365_236580


namespace half_hexagon_perimeter_l2365_236557

/-- A polygon that forms one half of a regular hexagon by symmetrically splitting it -/
structure HalfHexagonPolygon where
  side_length : ℝ
  is_positive : side_length > 0

/-- The perimeter of a HalfHexagonPolygon -/
def perimeter (p : HalfHexagonPolygon) : ℝ :=
  3 * p.side_length

/-- Theorem: The perimeter of a HalfHexagonPolygon is equal to 3 times its side length -/
theorem half_hexagon_perimeter (p : HalfHexagonPolygon) :
  perimeter p = 3 * p.side_length := by
  sorry

end half_hexagon_perimeter_l2365_236557


namespace expression_simplification_l2365_236524

theorem expression_simplification (x y : ℝ) :
  3*x + 4*y + 5*x^2 + 2 - (8 - 5*x - 3*y - 2*x^2) = 7*x^2 + 8*x + 7*y - 6 := by
  sorry

end expression_simplification_l2365_236524


namespace adults_group_size_l2365_236533

/-- The number of children in each group -/
def children_per_group : ℕ := 15

/-- The minimum number of adults (and children) attending -/
def min_attendees : ℕ := 255

/-- The number of adults in each group -/
def adults_per_group : ℕ := 15

theorem adults_group_size :
  (min_attendees % children_per_group = 0) →
  (min_attendees % adults_per_group = 0) →
  (min_attendees / children_per_group = min_attendees / adults_per_group) →
  adults_per_group = 15 := by
  sorry

end adults_group_size_l2365_236533


namespace inequality_solution_l2365_236591

-- Define the inequality function
def f (x : ℝ) := x * (x - 2)

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x > 2}

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end inequality_solution_l2365_236591


namespace almond_butter_cookie_cost_difference_l2365_236586

/-- The cost difference per batch between almond butter cookies and peanut butter cookies -/
def cost_difference (peanut_butter_price : ℝ) (almond_butter_multiplier : ℝ) (jar_fraction : ℝ) (sugar_price_difference : ℝ) : ℝ :=
  (almond_butter_multiplier * peanut_butter_price * jar_fraction - peanut_butter_price * jar_fraction) + sugar_price_difference

/-- Theorem: The cost difference per batch between almond butter cookies and peanut butter cookies is $3.50 -/
theorem almond_butter_cookie_cost_difference :
  cost_difference 3 3 (1/2) 0.5 = 3.5 := by
  sorry

end almond_butter_cookie_cost_difference_l2365_236586


namespace backpack_solution_l2365_236547

/-- Represents the prices and quantities of backpacks -/
structure BackpackData where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- Conditions for the backpack problem -/
def backpack_conditions (d : BackpackData) : Prop :=
  d.price_a = 2 * d.price_b - 30 ∧
  2 * d.price_a + 3 * d.price_b = 255 ∧
  d.quantity_a + d.quantity_b = 200 ∧
  50 * d.quantity_a + 40 * d.quantity_b ≤ 8900 ∧
  d.quantity_a > 87

/-- The theorem stating the correct prices and possible purchasing plans -/
theorem backpack_solution :
  ∃ (d : BackpackData),
    backpack_conditions d ∧
    d.price_a = 60 ∧
    d.price_b = 45 ∧
    ((d.quantity_a = 88 ∧ d.quantity_b = 112) ∨
     (d.quantity_a = 89 ∧ d.quantity_b = 111) ∨
     (d.quantity_a = 90 ∧ d.quantity_b = 110)) :=
  sorry

end backpack_solution_l2365_236547


namespace village_population_l2365_236500

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) : 
  partial_population = 36000 → percentage = 9/10 → 
  (percentage * total_population : ℚ) = partial_population → 
  total_population = 40000 := by
sorry

end village_population_l2365_236500


namespace prob_spade_first_ace_last_value_l2365_236569

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of aces in a standard deck -/
def NumAces : ℕ := 4

/-- Probability of drawing three cards from a standard 52-card deck,
    where the first card is a spade and the last card is an ace -/
def prob_spade_first_ace_last : ℚ :=
  (NumSpades * NumAces + NumAces - 1) / (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2))

theorem prob_spade_first_ace_last_value :
  prob_spade_first_ace_last = 51 / 2600 := by
  sorry

end prob_spade_first_ace_last_value_l2365_236569


namespace dinner_time_l2365_236588

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_def : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

/-- The starting time (4:00 pm) -/
def startTime : Time := ⟨16, 0, sorry⟩

/-- The total duration of tasks in minutes -/
def totalTaskDuration : ℕ := 30 + 30 + 10 + 20 + 90

/-- Theorem: Adding the total task duration to the start time results in 7:00 pm -/
theorem dinner_time : addMinutes startTime totalTaskDuration = ⟨19, 0, sorry⟩ := by
  sorry

end dinner_time_l2365_236588


namespace lcm_gcf_ratio_252_630_l2365_236510

theorem lcm_gcf_ratio_252_630 : Nat.lcm 252 630 / Nat.gcd 252 630 = 10 := by
  sorry

end lcm_gcf_ratio_252_630_l2365_236510


namespace tan_2theta_value_l2365_236597

theorem tan_2theta_value (θ : Real) (h1 : θ ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin θ - Real.cos θ = Real.sqrt 5 / 5) : 
  Real.tan (2 * θ) = -4 / 3 := by
  sorry

end tan_2theta_value_l2365_236597


namespace greatest_integer_satisfying_inequality_l2365_236505

theorem greatest_integer_satisfying_inequality :
  ∀ n : ℤ, (∀ x : ℤ, 3 * |2 * x + 1| + 10 > 28 → x ≤ n) → n = 2 := by
  sorry

end greatest_integer_satisfying_inequality_l2365_236505


namespace sugar_solution_percentage_l2365_236503

theorem sugar_solution_percentage (x : ℝ) :
  (3/4 * x + 1/4 * 40 = 16) → x = 8 := by
  sorry

end sugar_solution_percentage_l2365_236503


namespace x_one_value_l2365_236558

theorem x_one_value (x₁ x₂ x₃ : ℝ) 
  (h_order : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_sum : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/4) : 
  x₁ = 3/4 := by
  sorry

end x_one_value_l2365_236558


namespace square_sum_given_sum_and_product_l2365_236598

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -6) : 
  x^2 + y^2 = 16 := by
sorry

end square_sum_given_sum_and_product_l2365_236598


namespace fraction_equivalence_l2365_236590

theorem fraction_equivalence : 
  (14 / 10 : ℚ) = 7 / 5 ∧ 
  (1 + 2 / 5 : ℚ) = 7 / 5 ∧ 
  (1 + 7 / 25 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 2 / 10 : ℚ) ≠ 7 / 5 ∧ 
  (1 + 14 / 70 : ℚ) ≠ 7 / 5 := by
  sorry

end fraction_equivalence_l2365_236590


namespace inequality_proof_l2365_236530

theorem inequality_proof (a b c : ℝ) : 
  a = Real.log 5 - Real.log 3 →
  b = (2 / 5) * Real.exp (2 / 3) →
  c = 2 / 3 →
  b > c ∧ c > a := by
  sorry

end inequality_proof_l2365_236530


namespace hannah_trip_cost_l2365_236559

/-- Calculates the cost of gas for a trip given odometer readings, fuel efficiency, and gas price -/
def trip_gas_cost (initial_reading : ℕ) (final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  ((final_reading - initial_reading : ℚ) / fuel_efficiency) * gas_price

theorem hannah_trip_cost :
  let initial_reading : ℕ := 32150
  let final_reading : ℕ := 32178
  let fuel_efficiency : ℚ := 25
  let gas_price : ℚ := 375/100
  trip_gas_cost initial_reading final_reading fuel_efficiency gas_price = 420/100 := by
  sorry

end hannah_trip_cost_l2365_236559


namespace difference_of_squares_262_258_l2365_236545

theorem difference_of_squares_262_258 : 262^2 - 258^2 = 2080 := by
  sorry

end difference_of_squares_262_258_l2365_236545


namespace total_sibling_age_l2365_236561

/-- Represents the ages of four siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ

/-- Theorem stating the total age of the siblings -/
theorem total_sibling_age (ages : SiblingAges) : 
  ages.susan = 15 → 
  ages.bob = 11 → 
  ages.arthur = ages.susan + 2 → 
  ages.tom = ages.bob - 3 → 
  ages.susan + ages.arthur + ages.tom + ages.bob = 51 := by
  sorry

end total_sibling_age_l2365_236561


namespace no_such_function_l2365_236572

theorem no_such_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end no_such_function_l2365_236572


namespace shopkeeper_profit_loss_l2365_236541

theorem shopkeeper_profit_loss (cost : ℝ) : 
  cost > 0 →
  let profit_percent := 10
  let loss_percent := 10
  let selling_price1 := cost * (1 + profit_percent / 100)
  let selling_price2 := cost * (1 - loss_percent / 100)
  let total_cost := 2 * cost
  let total_selling_price := selling_price1 + selling_price2
  (total_selling_price - total_cost) / total_cost * 100 = 0 :=
by sorry

end shopkeeper_profit_loss_l2365_236541


namespace point_symmetry_l2365_236512

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two points are symmetric with respect to a line -/
def isSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  -- The midpoint of the two points lies on the line
  l.a * ((p1.x + p2.x) / 2) + l.b * ((p1.y + p2.y) / 2) + l.c = 0 ∧
  -- The line connecting the two points is perpendicular to the given line
  (p2.x - p1.x) * l.a + (p2.y - p1.y) * l.b = 0

theorem point_symmetry :
  let a : Point := ⟨-1, 2⟩
  let b : Point := ⟨1, 4⟩
  let l : Line := ⟨1, 1, -3⟩
  isSymmetric a b l := by sorry

end point_symmetry_l2365_236512


namespace largest_root_of_cubic_l2365_236587

theorem largest_root_of_cubic (p q r : ℝ) : 
  p + q + r = 3 → 
  p * q + p * r + q * r = -8 → 
  p * q * r = -18 → 
  ∃ (x : ℝ), x = Real.sqrt 6 ∧ 
    x = max p (max q r) ∧
    x^3 - 3*x^2 - 8*x + 18 = 0 := by
sorry

end largest_root_of_cubic_l2365_236587


namespace line_bisects_circle_l2365_236506

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem line_bisects_circle :
  line_eq (circle_center.1) (circle_center.2) ∧
  ∃ (r : ℝ), ∀ (x y : ℝ), circle_eq x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2 :=
sorry

end line_bisects_circle_l2365_236506


namespace students_taking_one_subject_l2365_236592

theorem students_taking_one_subject (both : ℕ) (math : ℕ) (only_science : ℕ)
  (h1 : both = 15)
  (h2 : math = 30)
  (h3 : only_science = 18) :
  math - both + only_science = 33 :=
by sorry

end students_taking_one_subject_l2365_236592


namespace congruence_mod_210_l2365_236520

theorem congruence_mod_210 (x : ℤ) : x^5 ≡ x [ZMOD 210] ↔ x ≡ 0 [ZMOD 7] ∨ x ≡ 1 [ZMOD 7] ∨ x ≡ -1 [ZMOD 7] := by
  sorry

end congruence_mod_210_l2365_236520


namespace junk_food_ratio_l2365_236570

theorem junk_food_ratio (weekly_allowance sweets_cost savings : ℚ)
  (h1 : weekly_allowance = 30)
  (h2 : sweets_cost = 8)
  (h3 : savings = 12)
  (h4 : weekly_allowance = sweets_cost + savings + (weekly_allowance - sweets_cost - savings)) :
  (weekly_allowance - sweets_cost - savings) / weekly_allowance = 1 / 3 := by
  sorry

end junk_food_ratio_l2365_236570


namespace circplus_assoc_l2365_236552

/-- The custom operation ⊕ on real numbers -/
def circplus (x y : ℝ) : ℝ := x + y - x * y

/-- Theorem stating that the ⊕ operation is associative -/
theorem circplus_assoc :
  ∀ (x y z : ℝ), circplus (circplus x y) z = circplus x (circplus y z) := by
  sorry

end circplus_assoc_l2365_236552


namespace right_triangle_has_one_right_angle_l2365_236527

/-- A right triangle is a triangle with one right angle -/
structure RightTriangle where
  /-- The measure of the right angle in degrees -/
  right_angle : ℝ
  /-- The right angle measures 90 degrees -/
  is_right : right_angle = 90

/-- The number of right angles in a right triangle -/
def num_right_angles (t : RightTriangle) : ℕ := 1

theorem right_triangle_has_one_right_angle (t : RightTriangle) : 
  num_right_angles t = 1 := by
  sorry

#check right_triangle_has_one_right_angle

end right_triangle_has_one_right_angle_l2365_236527


namespace recipe_measurements_l2365_236565

/- Define the required amounts and cup capacities -/
def required_flour : ℚ := 15/4  -- 3¾ cups
def required_milk : ℚ := 3/2    -- 1½ cups
def flour_cup_capacity : ℚ := 1/3
def milk_cup_capacity : ℚ := 1/4

/- Define the number of fills for each ingredient -/
def flour_fills : ℕ := 12
def milk_fills : ℕ := 6

/- Theorem statement -/
theorem recipe_measurements :
  (↑flour_fills * flour_cup_capacity ≥ required_flour) ∧
  ((↑flour_fills - 1) * flour_cup_capacity < required_flour) ∧
  (↑milk_fills * milk_cup_capacity = required_milk) :=
by sorry

end recipe_measurements_l2365_236565


namespace passes_through_first_and_third_quadrants_l2365_236522

def proportional_function (x : ℝ) : ℝ := x

theorem passes_through_first_and_third_quadrants :
  (∀ x : ℝ, x > 0 → proportional_function x > 0) ∧
  (∀ x : ℝ, x < 0 → proportional_function x < 0) :=
by sorry

end passes_through_first_and_third_quadrants_l2365_236522


namespace power_function_through_point_l2365_236560

/-- A power function passing through (4, 1/2) has f(1/16) = 4 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x^a) →  -- f is a power function
  f 4 = 1/2 →             -- f passes through (4, 1/2)
  f (1/16) = 4 :=         -- prove f(1/16) = 4
by sorry

end power_function_through_point_l2365_236560


namespace margo_travel_distance_l2365_236562

/-- The total distance traveled by Margo -/
def total_distance (bicycle_time walk_time average_rate : ℚ) : ℚ :=
  average_rate * (bicycle_time + walk_time) / 60

/-- Theorem: Given the conditions, Margo traveled 4 miles -/
theorem margo_travel_distance :
  let bicycle_time : ℚ := 15
  let walk_time : ℚ := 25
  let average_rate : ℚ := 6
  total_distance bicycle_time walk_time average_rate = 4 := by
  sorry

end margo_travel_distance_l2365_236562


namespace compound_weight_l2365_236563

-- Define the atomic weights
def atomic_weight_H : ℝ := 1
def atomic_weight_Cl : ℝ := 35.5
def atomic_weight_O : ℝ := 16

-- Define the total molecular weight
def total_weight : ℝ := 68

-- Define the number of oxygen atoms
def n : ℕ := 2

-- Theorem statement
theorem compound_weight :
  atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O = total_weight :=
by sorry

end compound_weight_l2365_236563


namespace parabola_reflection_translation_l2365_236578

theorem parabola_reflection_translation (a b c : ℝ) :
  let f := fun x => a * (x - 3)^2 + b * (x - 3) + c
  let g := fun x => -a * (x + 3)^2 - b * (x + 3) - c
  ∀ x, (f + g) x = -12 * a * x - 6 * b := by sorry

end parabola_reflection_translation_l2365_236578


namespace jerry_remaining_money_l2365_236517

/-- Calculates the remaining money after spending -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem: Jerry's remaining money is $12 -/
theorem jerry_remaining_money :
  remaining_money 18 6 = 12 := by
  sorry

end jerry_remaining_money_l2365_236517


namespace red_white_flowers_l2365_236539

/-- Represents the number of flowers of each color combination --/
structure FlowerCounts where
  total : ℕ
  yellowWhite : ℕ
  redYellow : ℕ
  redWhite : ℕ

/-- The difference between flowers containing red and white --/
def redWhiteDifference (f : FlowerCounts) : ℤ :=
  (f.redYellow + f.redWhite : ℤ) - (f.yellowWhite + f.redWhite : ℤ)

/-- Theorem stating the number of red and white flowers --/
theorem red_white_flowers (f : FlowerCounts) 
  (h_total : f.total = 44)
  (h_yellowWhite : f.yellowWhite = 13)
  (h_redYellow : f.redYellow = 17)
  (h_redWhiteDiff : redWhiteDifference f = 4) :
  f.redWhite = 14 := by
  sorry

end red_white_flowers_l2365_236539


namespace equation_solution_l2365_236529

theorem equation_solution : ∃! x : ℝ, 5 * 5^x + Real.sqrt (25 * 25^x) = 50 := by
  sorry

end equation_solution_l2365_236529


namespace pattern_equality_l2365_236528

theorem pattern_equality (n : ℕ+) : (2*n + 2)^2 - 4*n^2 = 8*n + 4 := by
  sorry

end pattern_equality_l2365_236528


namespace bear_population_difference_l2365_236596

theorem bear_population_difference :
  ∀ (white black brown : ℕ),
    black = 2 * white →
    black = 60 →
    white + black + brown = 190 →
    brown - black = 40 :=
by
  sorry

end bear_population_difference_l2365_236596


namespace problem_solution_l2365_236589

theorem problem_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 20) : x - y = 5/2 := by
  sorry

end problem_solution_l2365_236589


namespace mason_daily_water_l2365_236583

/-- The number of cups of water Theo drinks per day -/
def theo_daily : ℕ := 8

/-- The number of cups of water Roxy drinks per day -/
def roxy_daily : ℕ := 9

/-- The total number of cups of water the siblings drink in one week -/
def total_weekly : ℕ := 168

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Proves that Mason drinks 7 cups of water every day -/
theorem mason_daily_water : ℕ := by
  sorry

end mason_daily_water_l2365_236583


namespace ship_food_supply_l2365_236594

/-- Calculates the remaining food supply on a ship after a specific consumption pattern. -/
theorem ship_food_supply (initial_supply : ℝ) : 
  initial_supply = 400 →
  (initial_supply - 2/5 * initial_supply) - 3/5 * (initial_supply - 2/5 * initial_supply) = 96 := by
  sorry

#check ship_food_supply

end ship_food_supply_l2365_236594


namespace problem_statement_l2365_236573

theorem problem_statement : 
  (∀ x : ℝ, ∀ a : ℝ, x^2 + a*x + a^2 ≥ 0) ∨ (∃ x₀ : ℕ+, 2 * (x₀.val)^2 - 1 ≤ 0) :=
by sorry

end problem_statement_l2365_236573


namespace number_equation_solution_l2365_236525

theorem number_equation_solution : ∃! x : ℝ, x + 2 + 8 = 3 * x := by
  sorry

end number_equation_solution_l2365_236525


namespace angle_between_v_and_w_l2365_236584

/-- The angle between two vectors in ℝ³ -/
def angle (v w : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Vector 1 -/
def v : ℝ × ℝ × ℝ := (3, -2, 2)

/-- Vector 2 -/
def w : ℝ × ℝ × ℝ := (2, 2, -1)

/-- Theorem: The angle between vectors v and w is 90° -/
theorem angle_between_v_and_w : angle v w = 90 := by sorry

end angle_between_v_and_w_l2365_236584


namespace sum_four_pentagons_l2365_236576

/-- The value of a square -/
def square : ℚ := sorry

/-- The value of a pentagon -/
def pentagon : ℚ := sorry

/-- First equation: square + 3*pentagon + square + pentagon = 25 -/
axiom eq1 : square + 3*pentagon + square + pentagon = 25

/-- Second equation: pentagon + 2*square + pentagon + square + pentagon = 22 -/
axiom eq2 : pentagon + 2*square + pentagon + square + pentagon = 22

/-- The sum of four pentagons is equal to 62/3 -/
theorem sum_four_pentagons : 4 * pentagon = 62/3 := by sorry

end sum_four_pentagons_l2365_236576


namespace gcd_problem_l2365_236534

theorem gcd_problem (b : ℤ) (h : 570 ∣ b) : Int.gcd (4*b^3 + b^2 + 5*b + 95) b = 95 := by
  sorry

end gcd_problem_l2365_236534


namespace cube_split_31_l2365_236555

/-- 
Given a natural number m > 1, returns the sequence of consecutive odd numbers 
that sum to m^3, starting from 2m - 1
-/
def cubeOddSequence (m : ℕ) : List ℕ := sorry

/-- 
Theorem: If 31 is in the sequence of odd numbers that sum to m^3 for m > 1, 
then m = 6
-/
theorem cube_split_31 (m : ℕ) (h1 : m > 1) : 
  31 ∈ cubeOddSequence m → m = 6 := by sorry

end cube_split_31_l2365_236555


namespace circle_c_value_l2365_236553

/-- The circle equation with parameter c -/
def circle_equation (x y c : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 6*y + c = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-4, 3)

/-- The radius of the circle -/
def circle_radius : ℝ := 5

/-- Theorem: The value of c in the circle equation is 0 -/
theorem circle_c_value : ∃ (c : ℝ), 
  (∀ (x y : ℝ), circle_equation x y c ↔ 
    ((x + 4)^2 + (y - 3)^2 = circle_radius^2)) ∧ 
  c = 0 := by
  sorry

end circle_c_value_l2365_236553


namespace f_symmetry_about_y_axis_l2365_236535

def f (x : ℝ) : ℝ := |x|

theorem f_symmetry_about_y_axis : ∀ x : ℝ, f x = f (-x) := by
  sorry

end f_symmetry_about_y_axis_l2365_236535


namespace adjacent_diff_one_l2365_236519

/-- Represents a 9x9 table filled with integers from 1 to 81 --/
def Table := Fin 9 → Fin 9 → Fin 81

/-- Two cells are adjacent if they are horizontally or vertically neighboring --/
def adjacent (i j i' j' : Fin 9) : Prop :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j'.val + 1 = j.val)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i'.val + 1 = i.val))

/-- The table contains all integers from 1 to 81 exactly once --/
def validTable (t : Table) : Prop :=
  ∀ n : Fin 81, ∃! (i j : Fin 9), t i j = n

/-- Main theorem: In a 9x9 table filled with integers from 1 to 81,
    there exist two adjacent cells whose values differ by exactly 1 --/
theorem adjacent_diff_one (t : Table) (h : validTable t) :
  ∃ (i j i' j' : Fin 9), adjacent i j i' j' ∧ 
    (t i j).val = (t i' j').val + 1 ∨ (t i j).val + 1 = (t i' j').val :=
sorry

end adjacent_diff_one_l2365_236519


namespace sugar_weight_loss_fraction_l2365_236523

theorem sugar_weight_loss_fraction (green_beans_weight sugar_weight rice_weight remaining_weight : ℝ) :
  green_beans_weight = 60 →
  rice_weight = green_beans_weight - 30 →
  sugar_weight = green_beans_weight - 10 →
  remaining_weight = 120 →
  (green_beans_weight + (2/3 * rice_weight) + sugar_weight - remaining_weight) / sugar_weight = 1/5 := by
  sorry

end sugar_weight_loss_fraction_l2365_236523


namespace smallest_c_l2365_236593

/-- A square with side length c -/
structure Square (c : ℝ) where
  side : c > 0

/-- A coloring of points on a square -/
def Coloring (c : ℝ) := Square c → Bool

/-- The distance between two points on a square -/
def distance (c : ℝ) (p q : Square c) : ℝ := sorry

/-- There exist two points of the same color with distance at least √5 -/
def hasMonochromaticPair (c : ℝ) (coloring : Coloring c) : Prop :=
  ∃ (p q : Square c), coloring p = coloring q ∧ distance c p q ≥ Real.sqrt 5

/-- The smallest possible value of c satisfying the condition -/
theorem smallest_c : 
  (∀ c : ℝ, c ≥ Real.sqrt 10 / 2 → ∀ coloring : Coloring c, hasMonochromaticPair c coloring) ∧
  (∀ c : ℝ, c < Real.sqrt 10 / 2 → ∃ coloring : Coloring c, ¬hasMonochromaticPair c coloring) :=
sorry

end smallest_c_l2365_236593


namespace last_card_identifiable_determine_last_card_back_l2365_236540

/-- Represents a card with two sides -/
structure Card where
  front : ℕ
  back : ℕ

/-- Creates a deck of n cards -/
def create_deck (n : ℕ) : List Card :=
  List.range n |>.map (λ i => ⟨i, i + 1⟩)

/-- Checks if a number appears in a list -/
def appears_in (k : ℕ) (list : List ℕ) : Prop :=
  k ∈ list

/-- Theorem: Determine if the back of the last card can be identified -/
theorem last_card_identifiable (n : ℕ) (shown : List ℕ) (last : ℕ) : Prop :=
  let deck := create_deck n
  last = 0 ∨ last = n ∨
  (1 ≤ last ∧ last ≤ n - 1 ∧ (appears_in (last - 1) shown ∨ appears_in (last + 1) shown))

/-- Main theorem: Characterization of when the back of the last card can be determined -/
theorem determine_last_card_back (n : ℕ) (shown : List ℕ) (last : ℕ) :
  last_card_identifiable n shown last ↔
  ∃ (card : Card), card ∈ create_deck n ∧
    ((card.front = last ∧ ∃ k, k ∈ shown ∧ k = card.back) ∨
     (card.back = last ∧ ∃ k, k ∈ shown ∧ k = card.front)) :=
  sorry


end last_card_identifiable_determine_last_card_back_l2365_236540


namespace seven_digit_multiple_l2365_236548

theorem seven_digit_multiple : ∀ (A B C : ℕ),
  (A < 10 ∧ B < 10 ∧ C < 10) →
  (∃ (k₁ k₂ k₃ : ℕ), 
    25000000 + A * 100000 + B * 10000 + 3300 + C = 8 * k₁ ∧
    25000000 + A * 100000 + B * 10000 + 3300 + C = 9 * k₂ ∧
    25000000 + A * 100000 + B * 10000 + 3300 + C = 11 * k₃) →
  A + B + C = 14 := by
sorry

end seven_digit_multiple_l2365_236548


namespace square_plus_eight_divisible_by_eleven_l2365_236509

theorem square_plus_eight_divisible_by_eleven : 
  ∃ k : ℤ, 5^2 + 8 = 11 * k := by
  sorry

end square_plus_eight_divisible_by_eleven_l2365_236509


namespace work_completion_l2365_236526

theorem work_completion (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : 
  initial_days = 6 → absent_men = 4 → final_days = 12 →
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * final_days ∧
    original_men = 8 :=
by
  sorry

end work_completion_l2365_236526


namespace watch_cost_in_dollars_l2365_236501

/-- The cost of a watch in dollars when paid with dimes -/
def watch_cost (num_dimes : ℕ) (dime_value : ℚ) : ℚ :=
  (num_dimes : ℚ) * dime_value

/-- Theorem: If Greyson paid for a watch with 50 dimes, and each dime is worth $0.10, then the cost of the watch is $5.00 -/
theorem watch_cost_in_dollars :
  watch_cost 50 (1/10) = 5 :=
sorry

end watch_cost_in_dollars_l2365_236501


namespace planes_parallel_if_perpendicular_to_same_line_l2365_236543

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (non_overlapping : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : perpendicular m β)
  (h3 : non_overlapping α β) :
  parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l2365_236543
