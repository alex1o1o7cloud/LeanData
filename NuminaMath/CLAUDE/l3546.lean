import Mathlib

namespace number_difference_l3546_354693

theorem number_difference (a b : ℕ) (h1 : a + b = 27630) (h2 : 5 * a + 5 = b) :
  b - a = 18421 := by
  sorry

end number_difference_l3546_354693


namespace inequality1_solution_inequality2_solution_inequality3_solution_inequality4_no_solution_l3546_354605

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x^2 + 5*x + 6 < 0
def inequality2 (x : ℝ) : Prop := -x^2 + 9*x - 20 < 0
def inequality3 (x : ℝ) : Prop := x^2 + x - 56 < 0
def inequality4 (x : ℝ) : Prop := 9*x^2 + 4 < 12*x

-- State the theorems
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ -3 < x ∧ x < -2 := by sorry

theorem inequality2_solution : 
  ∀ x : ℝ, inequality2 x ↔ x < 4 ∨ x > 5 := by sorry

theorem inequality3_solution : 
  ∀ x : ℝ, inequality3 x ↔ -8 < x ∧ x < 7 := by sorry

theorem inequality4_no_solution : 
  ¬∃ x : ℝ, inequality4 x := by sorry

end inequality1_solution_inequality2_solution_inequality3_solution_inequality4_no_solution_l3546_354605


namespace triangle_equilateral_from_sequences_l3546_354672

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively is equilateral
if its angles form an arithmetic sequence and its sides form a geometric sequence. -/
theorem triangle_equilateral_from_sequences (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- positive angles
  A + B + C = π →  -- sum of angles in a triangle
  2 * B = A + C →  -- angles form arithmetic sequence
  b^2 = a * c →  -- sides form geometric sequence
  A = B ∧ B = C ∧ a = b ∧ b = c :=
by sorry

end triangle_equilateral_from_sequences_l3546_354672


namespace total_movies_is_74_l3546_354610

/-- Represents the number of movies watched by each person -/
structure MovieCounts where
  dalton : ℕ
  hunter : ℕ
  alex : ℕ
  bella : ℕ
  chris : ℕ

/-- Represents the number of movies watched by different groups -/
structure SharedMovies where
  all_five : ℕ
  dalton_hunter_alex : ℕ
  bella_chris : ℕ
  dalton_bella : ℕ
  alex_chris : ℕ

/-- Calculates the total number of different movies watched -/
def total_different_movies (individual : MovieCounts) (shared : SharedMovies) : ℕ :=
  individual.dalton + individual.hunter + individual.alex + individual.bella + individual.chris -
  (4 * shared.all_five + 2 * shared.dalton_hunter_alex + shared.bella_chris + shared.dalton_bella + shared.alex_chris)

/-- Theorem stating that the total number of different movies watched is 74 -/
theorem total_movies_is_74 (individual : MovieCounts) (shared : SharedMovies)
    (h1 : individual = ⟨20, 26, 35, 29, 16⟩)
    (h2 : shared = ⟨5, 4, 3, 2, 4⟩) :
    total_different_movies individual shared = 74 := by
  sorry

end total_movies_is_74_l3546_354610


namespace min_value_and_reciprocal_sum_l3546_354694

-- Define the function f
def f (a b c x : ℝ) : ℝ := |x + a| + |x - b| + c

-- State the theorem
theorem min_value_and_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hmin : ∀ x, f a b c x ≥ 5) 
  (hf_attains_min : ∃ x, f a b c x = 5) : 
  a + b + c = 5 ∧ (1/a + 1/b + 1/c ≥ 9/5 ∧ ∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 1/a' + 1/b' + 1/c' = 9/5) :=
sorry

end min_value_and_reciprocal_sum_l3546_354694


namespace specific_right_triangle_l3546_354614

/-- A right triangle with inscribed and circumscribed circles -/
structure RightTriangle where
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The radius of the circumscribed circle -/
  circumradius : ℝ
  /-- The shortest side of the triangle -/
  a : ℝ
  /-- The middle-length side of the triangle -/
  b : ℝ
  /-- The hypotenuse of the triangle -/
  c : ℝ
  /-- The triangle is right-angled -/
  right_angle : a^2 + b^2 = c^2
  /-- The inradius is correct for this triangle -/
  inradius_correct : inradius = (a + b - c) / 2
  /-- The circumradius is correct for this triangle -/
  circumradius_correct : circumradius = c / 2

/-- The main theorem about the specific right triangle -/
theorem specific_right_triangle :
  ∃ (t : RightTriangle), t.inradius = 8 ∧ t.circumradius = 41 ∧ t.a = 18 ∧ t.b = 80 ∧ t.c = 82 := by
  sorry

end specific_right_triangle_l3546_354614


namespace tan_function_property_l3546_354681

theorem tan_function_property (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π / 2))) → 
  a * Real.tan (b * π / 8) = 4 → 
  a * b = 8 := by sorry

end tan_function_property_l3546_354681


namespace inequality_proof_l3546_354692

theorem inequality_proof (f : ℝ → ℝ) (a m n : ℝ) :
  (∀ x, f x = |x + a|) →
  (Set.Icc (-9 : ℝ) 1 = {x | f x ≤ 5}) →
  (m > 0) →
  (n > 0) →
  (1/m + 1/(2*n) = a) →
  m + 2*n ≥ 1 := by
  sorry

end inequality_proof_l3546_354692


namespace license_plate_count_l3546_354617

/-- The number of consonants in the alphabet (excluding Y) -/
def num_consonants : ℕ := 20

/-- The number of vowels (including Y) -/
def num_vowels : ℕ := 6

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible license plates -/
def num_license_plates : ℕ := num_consonants * num_digits * num_vowels * (num_consonants - 1)

/-- Theorem stating the number of possible license plates -/
theorem license_plate_count : num_license_plates = 22800 := by
  sorry

end license_plate_count_l3546_354617


namespace tv_price_change_l3546_354629

theorem tv_price_change (x : ℝ) : 
  (100 - x) * 1.5 = 120 → x = 20 := by sorry

end tv_price_change_l3546_354629


namespace arithmetic_calculation_l3546_354682

theorem arithmetic_calculation : 15 * (1/3) + 45 * (2/3) = 35 := by
  sorry

end arithmetic_calculation_l3546_354682


namespace r_profit_share_l3546_354623

/-- Represents the profit share of a partner in a business partnership --/
def ProfitShare (initial_ratio : ℚ) (months_full : ℕ) (months_reduced : ℕ) (reduction_factor : ℚ) (total_profit : ℚ) : ℚ :=
  let total_investment := 12 * (4 + 6 + 10)
  let partner_investment := initial_ratio * months_full + initial_ratio * reduction_factor * months_reduced
  (partner_investment / total_investment) * total_profit

theorem r_profit_share :
  let p_ratio : ℚ := 4
  let q_ratio : ℚ := 6
  let r_ratio : ℚ := 10
  let total_profit : ℚ := 4650
  ProfitShare r_ratio 12 0 1 total_profit = 2325 := by
  sorry

end r_profit_share_l3546_354623


namespace rectangles_in_5x5_array_l3546_354637

/-- The number of rectangles in a square array of dots -/
def rectangles_in_array (n : ℕ) : ℕ := (n.choose 2) * (n.choose 2)

/-- Theorem: In a 5x5 square array of dots, there are 100 different rectangles 
    with sides parallel to the grid that can be formed by connecting four dots -/
theorem rectangles_in_5x5_array : rectangles_in_array 5 = 100 := by
  sorry

end rectangles_in_5x5_array_l3546_354637


namespace sum_always_four_digits_l3546_354635

-- Define nonzero digits
def NonzeroDigit := { n : ℕ | 1 ≤ n ∧ n ≤ 9 }

-- Define the sum function
def sum_numbers (C D : NonzeroDigit) : ℕ :=
  3654 + (100 * C.val + 41) + (10 * D.val + 2) + 111

-- Theorem statement
theorem sum_always_four_digits (C D : NonzeroDigit) :
  ∃ n : ℕ, 1000 ≤ sum_numbers C D ∧ sum_numbers C D < 10000 := by
  sorry

end sum_always_four_digits_l3546_354635


namespace max_roses_for_budget_l3546_354665

/-- Represents the different rose purchasing options --/
inductive RoseOption
  | Individual
  | OneDozen
  | TwoDozen
  | Bulk

/-- Returns the cost of a given rose option --/
def cost (option : RoseOption) : Rat :=
  match option with
  | RoseOption.Individual => 730/100
  | RoseOption.OneDozen => 36
  | RoseOption.TwoDozen => 50
  | RoseOption.Bulk => 200

/-- Returns the number of roses for a given option --/
def roses (option : RoseOption) : Nat :=
  match option with
  | RoseOption.Individual => 1
  | RoseOption.OneDozen => 12
  | RoseOption.TwoDozen => 24
  | RoseOption.Bulk => 100

/-- Represents a purchase of roses --/
structure Purchase where
  individual : Nat
  oneDozen : Nat
  twoDozen : Nat
  bulk : Nat

/-- Calculates the total cost of a purchase --/
def totalCost (p : Purchase) : Rat :=
  p.individual * cost RoseOption.Individual +
  p.oneDozen * cost RoseOption.OneDozen +
  p.twoDozen * cost RoseOption.TwoDozen +
  p.bulk * cost RoseOption.Bulk

/-- Calculates the total number of roses in a purchase --/
def totalRoses (p : Purchase) : Nat :=
  p.individual * roses RoseOption.Individual +
  p.oneDozen * roses RoseOption.OneDozen +
  p.twoDozen * roses RoseOption.TwoDozen +
  p.bulk * roses RoseOption.Bulk

/-- The budget constraint --/
def budget : Rat := 680

/-- Theorem: The maximum number of roses that can be purchased for $680 is 328 --/
theorem max_roses_for_budget :
  ∃ (p : Purchase),
    totalCost p ≤ budget ∧
    totalRoses p = 328 ∧
    ∀ (q : Purchase), totalCost q ≤ budget → totalRoses q ≤ totalRoses p :=
by sorry


end max_roses_for_budget_l3546_354665


namespace ferris_wheel_capacity_ferris_wheel_capacity_proof_l3546_354663

theorem ferris_wheel_capacity (small_seats large_seats : ℕ) 
  (large_seat_capacity : ℕ) (total_large_capacity : ℕ) : Prop :=
  small_seats = 3 ∧ 
  large_seats = 7 ∧ 
  large_seat_capacity = 12 ∧
  total_large_capacity = 84 →
  ¬∃ (small_seat_capacity : ℕ), 
    ∀ (total_capacity : ℕ), 
      total_capacity = small_seats * small_seat_capacity + total_large_capacity

theorem ferris_wheel_capacity_proof : 
  ferris_wheel_capacity 3 7 12 84 := by
  sorry

end ferris_wheel_capacity_ferris_wheel_capacity_proof_l3546_354663


namespace divisor_power_difference_l3546_354652

theorem divisor_power_difference (k : ℕ) : 
  (15 ^ k : ℕ) ∣ 759325 → 3 ^ k - k ^ 3 = 2 := by
  sorry

end divisor_power_difference_l3546_354652


namespace sum_squared_l3546_354677

theorem sum_squared (x y : ℝ) (h1 : x * (x + y) = 30) (h2 : y * (x + y) = 60) :
  (x + y)^2 = 90 := by
sorry

end sum_squared_l3546_354677


namespace max_inequality_sqrt_sum_l3546_354667

theorem max_inequality_sqrt_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) :
  ∃ (m : ℝ), m = 2 + Real.sqrt 5 ∧ 
  (∀ (x : ℝ), Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1) > x → x ≤ m) ∧
  Real.sqrt (4*a + 1) + Real.sqrt (4*b + 1) + Real.sqrt (4*c + 1) > m :=
by sorry

end max_inequality_sqrt_sum_l3546_354667


namespace negative_cube_root_of_negative_eight_equals_two_l3546_354611

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem negative_cube_root_of_negative_eight_equals_two :
  -cubeRoot (-8) = 2 := by sorry

end negative_cube_root_of_negative_eight_equals_two_l3546_354611


namespace solution_range_l3546_354645

theorem solution_range (x : ℝ) :
  x ≥ 2 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2)) = 2) →
  x ∈ Set.Icc 11 18 :=
by sorry

end solution_range_l3546_354645


namespace complex_product_polar_form_l3546_354643

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the problem
theorem complex_product_polar_form :
  ∃ (r θ : ℝ), 
    r > 0 ∧ 
    0 ≤ θ ∧ 
    θ < 2 * Real.pi ∧
    (4 * cis (30 * Real.pi / 180)) * (-3 * cis (45 * Real.pi / 180)) = r * cis θ ∧
    r = 12 ∧
    θ = 255 * Real.pi / 180 := by
  sorry

end complex_product_polar_form_l3546_354643


namespace opposite_silver_is_black_l3546_354616

-- Define the colors
inductive Color
  | Yellow
  | Orange
  | Blue
  | Black
  | Silver
  | Pink

-- Define a face of the cube
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  left : Face
  right : Face

-- Define a view of the cube
structure CubeView where
  top : Face
  front : Face
  right : Face

-- Define the theorem
theorem opposite_silver_is_black (c : Cube) 
  (view1 view2 view3 : CubeView)
  (h1 : c.top.color = Color.Black ∧ 
        c.right.color = Color.Blue)
  (h2 : view1.top.color = Color.Black ∧ 
        view1.front.color = Color.Pink ∧ 
        view1.right.color = Color.Blue)
  (h3 : view2.top.color = Color.Black ∧ 
        view2.front.color = Color.Orange ∧ 
        view2.right.color = Color.Blue)
  (h4 : view3.top.color = Color.Black ∧ 
        view3.front.color = Color.Yellow ∧ 
        view3.right.color = Color.Blue)
  (h5 : c.bottom.color = Color.Silver) :
  c.top.color = Color.Black :=
sorry

end opposite_silver_is_black_l3546_354616


namespace abs_difference_given_product_and_sum_l3546_354613

theorem abs_difference_given_product_and_sum (a b : ℝ) 
  (h1 : a * b = 6) 
  (h2 : a + b = 7) : 
  |a - b| = 5 := by
sorry

end abs_difference_given_product_and_sum_l3546_354613


namespace art_department_probabilities_l3546_354631

/-- The number of male members in the student art department -/
def num_males : ℕ := 4

/-- The number of female members in the student art department -/
def num_females : ℕ := 3

/-- The total number of members in the student art department -/
def total_members : ℕ := num_males + num_females

/-- The number of members to be selected for the art performance event -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly one female member -/
def prob_one_female : ℚ := 4 / 7

/-- The probability of selecting a specific female member given a specific male member is selected -/
def prob_female_given_male : ℚ := 1 / 6

theorem art_department_probabilities :
  (prob_one_female = 4 / 7) ∧
  (prob_female_given_male = 1 / 6) := by
  sorry

#check art_department_probabilities

end art_department_probabilities_l3546_354631


namespace triangle_inequality_l3546_354641

theorem triangle_inequality (a b c p : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : p = (a + b + c) / 2) :
  Real.sqrt (p - a) + Real.sqrt (p - b) + Real.sqrt (p - c) ≤ Real.sqrt (3 * p) := by
sorry

end triangle_inequality_l3546_354641


namespace isosceles_right_triangle_area_l3546_354699

/-- Given an isosceles right triangle with squares on its sides, 
    prove that its area is 32 square units. -/
theorem isosceles_right_triangle_area 
  (a b c : ℝ) 
  (h_isosceles : a = b) 
  (h_right : a^2 + b^2 = c^2) 
  (h_leg_square : a^2 = 64) 
  (h_hypotenuse_square : c^2 = 256) : 
  (1/2) * a^2 = 32 := by
  sorry

#check isosceles_right_triangle_area

end isosceles_right_triangle_area_l3546_354699


namespace percent_relation_l3546_354649

theorem percent_relation (x y : ℝ) (h : 0.6 * (x - y) = 0.3 * (x + y)) : y = (1/3) * x := by
  sorry

end percent_relation_l3546_354649


namespace concert_hat_wearers_l3546_354647

theorem concert_hat_wearers (total_attendees : ℕ) 
  (women_fraction : ℚ) (women_hat_percentage : ℚ) (men_hat_percentage : ℚ) :
  total_attendees = 3000 →
  women_fraction = 2/3 →
  women_hat_percentage = 15/100 →
  men_hat_percentage = 12/100 →
  ↑(total_attendees * (women_fraction * women_hat_percentage + 
    (1 - women_fraction) * men_hat_percentage)) = (420 : ℚ) := by
  sorry

end concert_hat_wearers_l3546_354647


namespace dividing_line_b_range_l3546_354618

/-- Triangle ABC with vertices A(-1,0), B(1,0), and C(0,1) -/
structure Triangle where
  A : ℝ × ℝ := (-1, 0)
  B : ℝ × ℝ := (1, 0)
  C : ℝ × ℝ := (0, 1)

/-- Line y = ax + b that divides the triangle -/
structure DividingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0

/-- The line divides the triangle into two parts of equal area -/
def dividesEqualArea (t : Triangle) (l : DividingLine) : Prop := sorry

/-- The range of b values that satisfy the condition -/
def validRange : Set ℝ := Set.Ioo (1 - Real.sqrt 2 / 2) (1 / 2)

/-- Theorem stating the range of b values -/
theorem dividing_line_b_range (t : Triangle) (l : DividingLine) 
  (h : dividesEqualArea t l) : l.b ∈ validRange := by
  sorry

end dividing_line_b_range_l3546_354618


namespace one_totally_damaged_carton_l3546_354680

/-- Represents the milk delivery problem --/
structure MilkDelivery where
  normal_cartons : ℕ
  jars_per_carton : ℕ
  cartons_shortage : ℕ
  damaged_cartons : ℕ
  damaged_jars_per_carton : ℕ
  good_jars : ℕ

/-- Calculates the number of totally damaged cartons --/
def totally_damaged_cartons (md : MilkDelivery) : ℕ :=
  let total_cartons := md.normal_cartons - md.cartons_shortage
  let total_jars := total_cartons * md.jars_per_carton
  let partially_damaged_jars := md.damaged_cartons * md.damaged_jars_per_carton
  let undamaged_jars := total_jars - partially_damaged_jars
  let additional_damaged_jars := undamaged_jars - md.good_jars
  additional_damaged_jars / md.jars_per_carton

/-- Theorem stating that the number of totally damaged cartons is 1 --/
theorem one_totally_damaged_carton (md : MilkDelivery) 
    (h1 : md.normal_cartons = 50)
    (h2 : md.jars_per_carton = 20)
    (h3 : md.cartons_shortage = 20)
    (h4 : md.damaged_cartons = 5)
    (h5 : md.damaged_jars_per_carton = 3)
    (h6 : md.good_jars = 565) :
    totally_damaged_cartons md = 1 := by
  sorry

#eval totally_damaged_cartons ⟨50, 20, 20, 5, 3, 565⟩

end one_totally_damaged_carton_l3546_354680


namespace expression_bounds_l3546_354656

theorem expression_bounds (x y z w : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
                    Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ∧
  Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
  Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ≤ 4 :=
by sorry

end expression_bounds_l3546_354656


namespace calculation_proof_l3546_354661

theorem calculation_proof : (1 / 6 : ℚ) * (-6 : ℚ) / (-1 / 6 : ℚ) * 6 = 36 := by
  sorry

end calculation_proof_l3546_354661


namespace adjacent_even_numbers_l3546_354660

theorem adjacent_even_numbers (n : ℕ) : 
  Odd (2*n + 1) ∧ 
  Even (2*n) ∧ 
  Even (2*n + 2) ∧
  (2*n + 1) - 1 = 2*n ∧
  (2*n + 1) + 1 = 2*n + 2 := by
sorry

end adjacent_even_numbers_l3546_354660


namespace horner_v1_value_l3546_354679

/-- Horner's method for polynomial evaluation -/
def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = 0.5x^5 + 4x^4 - 3x^2 + x - 1 -/
def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

theorem horner_v1_value :
  let x : ℝ := 3
  let v0 : ℝ := 0.5
  let v1 : ℝ := horner_step v0 x 4
  v1 = 5.5 := by sorry

end horner_v1_value_l3546_354679


namespace largest_prime_check_l3546_354634

theorem largest_prime_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) → Nat.Prime n :=
sorry

end largest_prime_check_l3546_354634


namespace opposite_of_neg_2023_l3546_354640

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem stating that the opposite of -2023 is 2023. -/
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l3546_354640


namespace speed_conversion_correct_l3546_354674

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def speed_mps : ℝ := 19.445999999999998

/-- Converts speed from m/s to km/h -/
def convert_speed (s : ℝ) : ℝ := s * mps_to_kmph

theorem speed_conversion_correct : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0005 ∧ |convert_speed speed_mps - 70.006| < ε :=
sorry

end speed_conversion_correct_l3546_354674


namespace existence_of_reduction_sequence_l3546_354608

/-- The game operation: either multiply by 2 or remove the unit digit -/
inductive GameOperation
| multiply_by_two
| remove_unit_digit

/-- Apply a single game operation to a natural number -/
def apply_operation (n : ℕ) (op : GameOperation) : ℕ :=
  match op with
  | GameOperation.multiply_by_two => 2 * n
  | GameOperation.remove_unit_digit => n / 10

/-- Predicate to check if a sequence of operations reduces a number to 1 -/
def reduces_to_one (start : ℕ) (ops : List GameOperation) : Prop :=
  start ≠ 0 ∧ List.foldl apply_operation start ops = 1

/-- Theorem: For any non-zero natural number, there exists a sequence of operations that reduces it to 1 -/
theorem existence_of_reduction_sequence (n : ℕ) : 
  n ≠ 0 → ∃ (ops : List GameOperation), reduces_to_one n ops := by
  sorry


end existence_of_reduction_sequence_l3546_354608


namespace expression_evaluation_l3546_354664

-- Define the expression
def expression : ℚ := -(2^3) + 6/5 * (2/5)

-- Theorem stating the equality
theorem expression_evaluation : expression = -7 - 13/25 := by sorry

end expression_evaluation_l3546_354664


namespace mans_age_to_sons_age_ratio_l3546_354646

theorem mans_age_to_sons_age_ratio : 
  ∀ (sons_current_age mans_current_age : ℕ),
    sons_current_age = 26 →
    mans_current_age = sons_current_age + 28 →
    (mans_current_age + 2) / (sons_current_age + 2) = 2 := by
  sorry

end mans_age_to_sons_age_ratio_l3546_354646


namespace solve_chalk_problem_l3546_354622

def chalk_problem (siblings friends chalk_per_person lost_chalk : ℕ) : Prop :=
  let total_people : ℕ := siblings + friends
  let total_chalk_needed : ℕ := total_people * chalk_per_person
  let available_chalk : ℕ := total_chalk_needed - lost_chalk
  let mom_brought : ℕ := total_chalk_needed - available_chalk
  mom_brought = 2

theorem solve_chalk_problem :
  chalk_problem 4 3 3 2 := by sorry

end solve_chalk_problem_l3546_354622


namespace aquarium_visitors_l3546_354655

-- Define the constants
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group_size : ℕ := 10
def total_earnings : ℕ := 240

-- Define the function to calculate the number of people who only went to the aquarium
def people_only_aquarium : ℕ :=
  (total_earnings - group_size * (admission_fee + tour_fee)) / admission_fee

-- Theorem to prove
theorem aquarium_visitors :
  people_only_aquarium = 5 := by
  sorry

end aquarium_visitors_l3546_354655


namespace sine_value_given_tangent_and_point_l3546_354671

theorem sine_value_given_tangent_and_point (α : Real) (m : Real) :
  (∃ (x y : Real), x = m ∧ y = 9 ∧ x^2 + y^2 ≠ 0 ∧ Real.tan α = y / x) →
  Real.tan α = 3 / 4 →
  Real.sin α = 3 / 5 := by
  sorry

end sine_value_given_tangent_and_point_l3546_354671


namespace special_numbers_are_correct_l3546_354632

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_special (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 9999 ∧
  let ab := n / 100
  let cd := n % 100
  is_perfect_square (ab - cd) ∧
  is_perfect_square (ab + cd) ∧
  (ab - cd) ∣ (ab + cd) ∧
  (ab + cd) ∣ n

def special_numbers : Finset ℕ :=
  {0100, 0400, 0900, 1600, 2500, 3600, 4900, 6400, 8100, 0504, 2016, 4536, 8064}

theorem special_numbers_are_correct :
  ∀ n : ℕ, is_special n ↔ n ∈ special_numbers := by sorry

end special_numbers_are_correct_l3546_354632


namespace quarter_point_quadrilateral_area_is_3_plus_2root2_l3546_354619

/-- Regular octagon with apothem 2 -/
structure RegularOctagon :=
  (apothem : ℝ)
  (is_regular : apothem = 2)

/-- Quarter point on a side of the octagon -/
def quarter_point (O : RegularOctagon) (i : Fin 8) : ℝ × ℝ := sorry

/-- The area of the quadrilateral formed by connecting quarter points -/
def quarter_point_quadrilateral_area (O : RegularOctagon) : ℝ :=
  let Q1 := quarter_point O 0
  let Q3 := quarter_point O 2
  let Q5 := quarter_point O 4
  let Q7 := quarter_point O 6
  sorry -- Area calculation

/-- Theorem: The area of the quadrilateral formed by connecting
    the quarter points of every other side of a regular octagon
    with apothem 2 is 3 + 2√2 -/
theorem quarter_point_quadrilateral_area_is_3_plus_2root2 (O : RegularOctagon) :
  quarter_point_quadrilateral_area O = 3 + 2 * Real.sqrt 2 := by
  sorry

end quarter_point_quadrilateral_area_is_3_plus_2root2_l3546_354619


namespace ellipse_equation_from_conditions_l3546_354658

/-- An ellipse centered at the origin with axes aligned with the coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation_from_conditions 
  (e : Ellipse)
  (h_major_twice_minor : e.a = 2 * e.b)
  (h_point_on_ellipse : ellipse_equation e 4 1) :
  ∀ x y, ellipse_equation e x y ↔ x^2 / 20 + y^2 / 5 = 1 :=
by sorry

end ellipse_equation_from_conditions_l3546_354658


namespace fraction_integer_iff_p_6_or_28_l3546_354615

theorem fraction_integer_iff_p_6_or_28 (p : ℕ+) :
  (∃ (n : ℕ+), (4 * p + 28 : ℚ) / (3 * p - 7) = n) ↔ p = 6 ∨ p = 28 := by
  sorry

end fraction_integer_iff_p_6_or_28_l3546_354615


namespace inequality_solution_inequality_proof_l3546_354601

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem inequality_solution (x : ℝ) : 
  f (x - 1) + f (1 - x) ≤ 2 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

-- Theorem for part II
theorem inequality_proof (x a : ℝ) (h : a < 0) : 
  f (a * x) - a * f x ≥ f a := by sorry

end inequality_solution_inequality_proof_l3546_354601


namespace value_of_y_l3546_354628

theorem value_of_y (x y : ℝ) (h1 : x^(3*y) = 9) (h2 : x = 3) : y = 2/3 := by
  sorry

end value_of_y_l3546_354628


namespace grid_paths_path_count_l3546_354683

theorem grid_paths (n m : ℕ) : (n + m).choose n = (n + m).choose m := by sorry

theorem path_count : Nat.choose 9 4 = 126 := by sorry

end grid_paths_path_count_l3546_354683


namespace linear_regression_at_6_l3546_354675

/-- Linear regression equation -/
def linear_regression (b a x : ℝ) : ℝ := b * x + a

theorem linear_regression_at_6 (b a : ℝ) (h1 : linear_regression b a 4 = 50) (h2 : b = -2) :
  linear_regression b a 6 = 46 := by
  sorry

end linear_regression_at_6_l3546_354675


namespace second_digit_max_l3546_354612

def original_number : ℚ := 0.123456789

-- Function to change a digit to 9 and swap with the next digit
def change_and_swap (n : ℚ) (pos : ℕ) : ℚ := sorry

-- Function to get the maximum value after change and swap operations
def max_after_change_and_swap (n : ℚ) : ℚ := sorry

-- Theorem stating that changing the second digit gives the maximum value
theorem second_digit_max :
  change_and_swap original_number 2 = max_after_change_and_swap original_number :=
sorry

end second_digit_max_l3546_354612


namespace emerson_rowing_distance_l3546_354653

/-- The total distance covered by Emerson on his rowing trip -/
def total_distance (initial_distance : ℕ) (second_segment : ℕ) (final_segment : ℕ) : ℕ :=
  initial_distance + second_segment + final_segment

/-- Theorem stating that Emerson's total distance is 39 miles -/
theorem emerson_rowing_distance :
  total_distance 6 15 18 = 39 := by
  sorry

end emerson_rowing_distance_l3546_354653


namespace area_PQR_l3546_354657

/-- Triangle ABC with given side lengths and points M, N, P, Q, R as described -/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Point M on AB
  AM : ℝ
  MB : ℝ
  -- Point N on BC
  CN : ℝ
  NB : ℝ
  -- Conditions
  side_lengths : AB = 20 ∧ BC = 21 ∧ CA = 29
  M_ratio : AM / MB = 3 / 2
  N_ratio : CN / NB = 2
  -- Existence of points P, Q, R (not explicitly defined)
  P_exists : ∃ P : ℝ × ℝ, True  -- P on AC
  Q_exists : ∃ Q : ℝ × ℝ, True  -- Q on AC
  R_exists : ∃ R : ℝ × ℝ, True  -- R as intersection of MP and NQ
  MP_parallel_BC : True  -- MP is parallel to BC
  NQ_parallel_AB : True  -- NQ is parallel to AB

/-- The area of triangle PQR is 224/15 -/
theorem area_PQR (t : TriangleABC) : ∃ area_PQR : ℝ, area_PQR = 224/15 := by
  sorry

end area_PQR_l3546_354657


namespace train_length_l3546_354642

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 9 → speed * time * (1000 / 3600) = 180 :=
by
  sorry

end train_length_l3546_354642


namespace parallelogram_circumference_l3546_354630

/-- The circumference of a parallelogram with side lengths 18 and 12 is 60. -/
theorem parallelogram_circumference : ℝ → ℝ → ℝ → Prop :=
  fun a b c => (a = 18 ∧ b = 12) → c = 2 * (a + b) → c = 60

/-- Proof of the theorem -/
lemma prove_parallelogram_circumference : parallelogram_circumference 18 12 60 := by
  sorry

end parallelogram_circumference_l3546_354630


namespace functional_equation_implies_g_five_l3546_354626

/-- A function g: ℝ → ℝ satisfying g(xy) = g(x)g(y) for all real x and y, and g(1) = 2 -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x * y) = g x * g y) ∧ (g 1 = 2)

/-- If g satisfies the functional equation, then g(5) = 32 -/
theorem functional_equation_implies_g_five (g : ℝ → ℝ) :
  FunctionalEquation g → g 5 = 32 := by
  sorry


end functional_equation_implies_g_five_l3546_354626


namespace class_size_l3546_354654

/-- The number of students in a class, given information about their sports participation -/
theorem class_size (football : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : football = 26)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 11) :
  football + tennis - both + neither = 40 := by
  sorry

end class_size_l3546_354654


namespace geometric_sequence_ratio_l3546_354697

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem geometric_sequence_ratio 
  (a₁ : ℝ) (q : ℝ) (h1 : q > 0) 
  (h2 : (geometric_sequence a₁ q 3) * (geometric_sequence a₁ q 9) = 
        2 * (geometric_sequence a₁ q 5)^2) : 
  q = Real.sqrt 2 := by
sorry

end geometric_sequence_ratio_l3546_354697


namespace pentagram_star_angle_pentagram_star_angle_proof_l3546_354603

/-- The angle at each point of a regular pentagram formed by extending the sides of a regular pentagon inscribed in a circle is 216°. -/
theorem pentagram_star_angle : ℝ :=
  let regular_pentagon_external_angle : ℝ := 360 / 5
  let star_point_angle : ℝ := 360 - 2 * regular_pentagon_external_angle
  216

/-- Proof of the pentagram star angle theorem. -/
theorem pentagram_star_angle_proof : pentagram_star_angle = 216 := by
  sorry

end pentagram_star_angle_pentagram_star_angle_proof_l3546_354603


namespace ice_cube_volume_l3546_354644

theorem ice_cube_volume (V : ℝ) : 
  V > 0 → -- Assume the original volume is positive
  (1/4 * (1/4 * V)) = 0.2 → -- After two hours, the volume is 0.2 cubic inches
  V = 3.2 := by
sorry

end ice_cube_volume_l3546_354644


namespace parabola_decreasing_implies_m_bound_l3546_354650

/-- If the function y = -x^2 - 4mx + 1 is decreasing on the interval [2, +∞), then m ≥ -1. -/
theorem parabola_decreasing_implies_m_bound (m : ℝ) : 
  (∀ x ≥ 2, ∀ y > x, -y^2 - 4*m*y + 1 < -x^2 - 4*m*x + 1) → 
  m ≥ -1 :=
by sorry

end parabola_decreasing_implies_m_bound_l3546_354650


namespace connor_test_scores_l3546_354609

theorem connor_test_scores (test1 test2 test3 test4 : ℕ) : 
  test1 = 82 →
  test2 = 75 →
  test1 ≤ 100 ∧ test2 ≤ 100 ∧ test3 ≤ 100 ∧ test4 ≤ 100 →
  (test1 + test2 + test3 + test4) / 4 = 85 →
  (test3 = 83 ∧ test4 = 100) ∨ (test3 = 100 ∧ test4 = 83) :=
by sorry

end connor_test_scores_l3546_354609


namespace sqrt_four_equals_plus_minus_two_l3546_354670

theorem sqrt_four_equals_plus_minus_two : ∀ (x : ℝ), x^2 = 4 → x = 2 ∨ x = -2 := by
  sorry

end sqrt_four_equals_plus_minus_two_l3546_354670


namespace min_sum_perfect_squares_l3546_354676

theorem min_sum_perfect_squares (x y : ℤ) (h : x^2 - y^2 = 221) :
  ∃ (a b : ℤ), a^2 - b^2 = 221 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 229 :=
by sorry

end min_sum_perfect_squares_l3546_354676


namespace simon_makes_three_pies_l3546_354666

/-- The number of blueberry pies Simon can make -/
def blueberry_pies (own_berries nearby_berries berries_per_pie : ℕ) : ℕ :=
  (own_berries + nearby_berries) / berries_per_pie

/-- Proof that Simon can make 3 blueberry pies -/
theorem simon_makes_three_pies :
  blueberry_pies 100 200 100 = 3 := by
  sorry

end simon_makes_three_pies_l3546_354666


namespace arithmetic_sequence_common_difference_l3546_354684

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 3 + a 6 = 11)
  (h3 : a 5 + a 8 = 39) :
  ∃ d : ℝ, d = 7 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l3546_354684


namespace power_relation_l3546_354625

theorem power_relation (x : ℝ) (n : ℕ) (h : x^(2*n) = 3) : x^(4*n) = 9 := by
  sorry

end power_relation_l3546_354625


namespace ashley_stair_climbing_time_l3546_354669

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem ashley_stair_climbing_time :
  arithmetic_sequence_sum 30 10 4 = 180 := by
  sorry

end ashley_stair_climbing_time_l3546_354669


namespace ant_spider_minimum_distance_l3546_354633

/-- The minimum distance between an ant and a spider under specific conditions -/
theorem ant_spider_minimum_distance :
  let ant_position (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let spider_position (x : ℝ) : ℝ × ℝ := (2 * x - 1, 0)
  let distance (θ x : ℝ) : ℝ := Real.sqrt ((ant_position θ).1 - (spider_position x).1)^2 + ((ant_position θ).2 - (spider_position x).2)^2
  ∃ (θ : ℝ), ∀ (φ : ℝ), distance θ θ ≤ distance φ φ ∧ distance θ θ = Real.sqrt 14 / 4 :=
by sorry

end ant_spider_minimum_distance_l3546_354633


namespace wedge_product_formula_l3546_354688

/-- The wedge product of two 2D vectors -/
def wedge_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

/-- Theorem: The wedge product of two 2D vectors (a₁, a₂) and (b₁, b₂) is equal to a₁b₂ - a₂b₁ -/
theorem wedge_product_formula (a b : ℝ × ℝ) :
  wedge_product a b = a.1 * b.2 - a.2 * b.1 := by
  sorry

end wedge_product_formula_l3546_354688


namespace seating_arrangements_count_l3546_354604

/-- Represents the number of people in the group -/
def num_people : Nat := 5

/-- Represents the number of seats in the car -/
def num_seats : Nat := 5

/-- Represents the number of people who can drive (Mr. and Mrs. Lopez) -/
def num_drivers : Nat := 2

/-- Calculates the number of seating arrangements -/
def seating_arrangements : Nat :=
  num_drivers * (num_people - 1) * Nat.factorial (num_seats - 2)

/-- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_count :
  seating_arrangements = 48 := by sorry

end seating_arrangements_count_l3546_354604


namespace middle_group_frequency_l3546_354620

theorem middle_group_frequency 
  (n : ℕ) 
  (total_area : ℝ) 
  (middle_area : ℝ) 
  (sample_size : ℕ) 
  (h1 : n > 0) 
  (h2 : middle_area = (1 / 5) * (total_area - middle_area)) 
  (h3 : sample_size = 300) : 
  (middle_area / total_area) * sample_size = 50 := by
sorry

end middle_group_frequency_l3546_354620


namespace comic_book_collections_l3546_354690

/-- Kymbrea's initial comic book collection size -/
def kymbrea_initial : ℕ := 50

/-- Kymbrea's monthly comic book addition rate -/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection size -/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book addition rate -/
def lashawn_rate : ℕ := 8

/-- The number of months after which LaShawn's collection will be three times Kymbrea's -/
def months : ℕ := 130

theorem comic_book_collections :
  lashawn_initial + lashawn_rate * months = 3 * (kymbrea_initial + kymbrea_rate * months) :=
by sorry

end comic_book_collections_l3546_354690


namespace quadratic_sum_l3546_354659

/-- Given a quadratic function f(x) = 12x^2 + 144x + 1728, 
    prove that when written in the form a(x+b)^2+c, 
    the sum a+b+c equals 18. -/
theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (12 * x^2 + 144 * x + 1728 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 18) := by
  sorry

end quadratic_sum_l3546_354659


namespace simplify_power_expression_l3546_354638

theorem simplify_power_expression (x y : ℝ) : (3 * x^2 * y)^4 = 81 * x^8 * y^4 := by
  sorry

end simplify_power_expression_l3546_354638


namespace two_digit_number_property_l3546_354636

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  units : ℕ
  tens : ℕ
  unit_constraint : units < 10
  ten_constraint : tens < 10

/-- The property that adding 18 to a number results in its reverse -/
def ReversesWhenAdd18 (n : TwoDigitNumber) : Prop :=
  n.tens * 10 + n.units + 18 = n.units * 10 + n.tens

/-- The main theorem -/
theorem two_digit_number_property (n : TwoDigitNumber) 
  (h1 : n.units + n.tens = 8) 
  (h2 : ReversesWhenAdd18 n) : 
  n.units + n.tens = 8 ∧ n.units + 10 * n.tens + 18 = 10 * n.units + n.tens := by
  sorry

end two_digit_number_property_l3546_354636


namespace binomial_10_2_l3546_354685

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by
  sorry

end binomial_10_2_l3546_354685


namespace car_average_speed_l3546_354602

/-- Calculate the average speed of a car given its uphill and downhill speeds and distances --/
theorem car_average_speed (uphill_speed downhill_speed uphill_distance downhill_distance : ℝ) :
  uphill_speed = 30 →
  downhill_speed = 40 →
  uphill_distance = 100 →
  downhill_distance = 50 →
  let total_distance := uphill_distance + downhill_distance
  let uphill_time := uphill_distance / uphill_speed
  let downhill_time := downhill_distance / downhill_speed
  let total_time := uphill_time + downhill_time
  let average_speed := total_distance / total_time
  average_speed = 1800 / 55 := by
  sorry

#eval (1800 : ℚ) / 55

end car_average_speed_l3546_354602


namespace exact_exponent_equality_l3546_354687

theorem exact_exponent_equality (n k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ m : ℕ, (2^(2^n) + 1 = p^k * m) ∧ ¬(∃ l : ℕ, 2^(2^n) + 1 = p^(k+1) * l)) →
  (∃ m : ℕ, (2^(p-1) - 1 = p^k * m) ∧ ¬(∃ l : ℕ, 2^(p-1) - 1 = p^(k+1) * l)) :=
by sorry

end exact_exponent_equality_l3546_354687


namespace evelyn_winning_strategy_l3546_354662

/-- Represents a player in the game -/
inductive Player
| Odin
| Evelyn

/-- Represents the state of a box in the game -/
structure Box where
  value : ℕ
  isEmpty : Bool

/-- Represents the game state -/
structure GameState where
  boxes : List Box
  currentPlayer : Player

/-- Defines a valid move in the game -/
def isValidMove (player : Player) (oldValue newValue : ℕ) : Prop :=
  match player with
  | Player.Odin => newValue < oldValue ∧ Odd newValue
  | Player.Evelyn => newValue < oldValue ∧ Even newValue

/-- Defines the winning condition for Evelyn -/
def isEvelynWin (state : GameState) : Prop :=
  let k := state.boxes.length / 3
  (state.boxes.filter (fun b => b.value = 0)).length = k ∧
  (state.boxes.filter (fun b => b.value ≠ 0)).all (fun b => b.value = 1)

/-- Defines the winning condition for Odin -/
def isOdinWin (state : GameState) : Prop :=
  let k := state.boxes.length / 3
  (state.boxes.filter (fun b => b.value = 0)).length = k ∧
  ¬(state.boxes.filter (fun b => b.value ≠ 0)).all (fun b => b.value = 1)

/-- Theorem stating that Evelyn has a winning strategy for all k -/
theorem evelyn_winning_strategy (k : ℕ) (h : k > 0) :
  ∃ (strategy : GameState → ℕ → ℕ),
    ∀ (initialState : GameState),
      initialState.boxes.length = 3 * k →
      initialState.currentPlayer = Player.Odin →
      (initialState.boxes.all (fun b => b.isEmpty)) →
      (∃ (finalState : GameState),
        (finalState.boxes.all (fun b => ¬b.isEmpty)) ∧
        (isEvelynWin finalState ∨
         (¬∃ (move : ℕ → ℕ), isValidMove Player.Odin (move 0) (move 1)))) :=
sorry

end evelyn_winning_strategy_l3546_354662


namespace mayors_cocoa_powder_l3546_354639

theorem mayors_cocoa_powder (total_needed : ℕ) (still_needed : ℕ) (h1 : total_needed = 306) (h2 : still_needed = 47) :
  total_needed - still_needed = 259 := by
  sorry

end mayors_cocoa_powder_l3546_354639


namespace smallest_with_16_divisors_l3546_354627

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a number has exactly 16 positive divisors -/
def has_16_divisors (n : ℕ+) : Prop := num_divisors n = 16

theorem smallest_with_16_divisors :
  ∀ n : ℕ+, has_16_divisors n → n ≥ 120 ∧ has_16_divisors 120 := by sorry

end smallest_with_16_divisors_l3546_354627


namespace most_stable_athlete_l3546_354678

def athlete_variance (a b c d : ℝ) : Prop :=
  a = 0.5 ∧ b = 0.5 ∧ c = 0.6 ∧ d = 0.4

theorem most_stable_athlete (a b c d : ℝ) 
  (h : athlete_variance a b c d) : 
  d < a ∧ d < b ∧ d < c :=
by
  sorry

#check most_stable_athlete

end most_stable_athlete_l3546_354678


namespace solution_set_is_open_interval_l3546_354668

open Set Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, (x - 1) * (deriv f x - f x) > 0)
variable (h3 : ∀ x, f (2 - x) = f x * exp (2 - 2*x))

-- Define the solution set
def solution_set := {x : ℝ | exp 2 * f (log x) < x * f 2}

-- State the theorem
theorem solution_set_is_open_interval :
  solution_set f = Ioo 1 (exp 2) :=
sorry

end solution_set_is_open_interval_l3546_354668


namespace train_speed_calculation_l3546_354648

/-- Given a train of length 120 m crossing a bridge of length 255 m in 30 seconds,
    prove that the speed of the train is 45 km/hr. -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_calculation_l3546_354648


namespace A_minus_2B_general_A_minus_2B_specific_l3546_354607

-- Define the algebraic expressions A and B
def A (x y : ℝ) : ℝ := 3 * x^2 - 5 * x * y - 2 * y^2
def B (x y : ℝ) : ℝ := x^2 - 3 * y

-- Theorem for part 1
theorem A_minus_2B_general (x y : ℝ) : 
  A x y - 2 * B x y = x^2 - 5 * x * y - 2 * y^2 + 6 * y := by sorry

-- Theorem for part 2
theorem A_minus_2B_specific : 
  A 2 (-1) - 2 * B 2 (-1) = 6 := by sorry

end A_minus_2B_general_A_minus_2B_specific_l3546_354607


namespace fraction_change_l3546_354695

/-- Given a fraction that changes from 1/12 to 2/15 when its numerator is increased by 20% and
    its denominator is decreased by x%, prove that x = 25. -/
theorem fraction_change (x : ℚ) : 
  (1 : ℚ) / 12 * (120 / 100) / ((100 - x) / 100) = 2 / 15 → x = 25 := by
  sorry

end fraction_change_l3546_354695


namespace f_monotone_decreasing_range_l3546_354673

/-- A piecewise function f dependent on parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 2)*x + 6*a - 1 else a^x

/-- Theorem stating the range of a for which f is monotonically decreasing -/
theorem f_monotone_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) → 3/8 ≤ a ∧ a < 2/3 :=
by sorry

end f_monotone_decreasing_range_l3546_354673


namespace clock_angle_at_8_clock_angle_at_8_is_120_l3546_354624

/-- The angle between clock hands at 8:00 -/
theorem clock_angle_at_8 : ℝ :=
  let total_degrees : ℝ := 360
  let hours_on_clock : ℕ := 12
  let current_hour : ℕ := 8
  let degrees_per_hour : ℝ := total_degrees / hours_on_clock
  let hour_hand_angle : ℝ := degrees_per_hour * current_hour
  let minute_hand_angle : ℝ := 0
  let angle_diff : ℝ := |hour_hand_angle - minute_hand_angle|
  min angle_diff (total_degrees - angle_diff)

/-- Theorem: The smaller angle between the hour-hand and minute-hand of a clock at 8:00 is 120° -/
theorem clock_angle_at_8_is_120 : clock_angle_at_8 = 120 := by
  sorry

end clock_angle_at_8_clock_angle_at_8_is_120_l3546_354624


namespace set_A_characterization_intersection_A_B_complement_A_union_B_l3546_354686

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) / (x + 1) > 0}
def B : Set ℝ := {x | x ≤ 4}

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Theorem statements
theorem set_A_characterization : A = {x | x > 3 ∨ x < -1} := by sorry

theorem intersection_A_B : A ∩ B = {x | 3 < x ∧ x ≤ 4} := by sorry

theorem complement_A_union_B : (Set.compl A) ∪ B = {x | x ≤ 4} := by sorry

end set_A_characterization_intersection_A_B_complement_A_union_B_l3546_354686


namespace expression_evaluation_l3546_354606

theorem expression_evaluation : 
  (0.86 : ℝ)^3 - (0.1 : ℝ)^3 / (0.86 : ℝ)^2 + 0.086 + (0.1 : ℝ)^2 = 0.730704 := by
  sorry

end expression_evaluation_l3546_354606


namespace sector_central_angle_l3546_354698

theorem sector_central_angle (area : ℝ) (perimeter : ℝ) (r : ℝ) (l : ℝ) (α : ℝ) :
  area = 1 →
  perimeter = 4 →
  2 * r + l = perimeter →
  area = 1/2 * l * r →
  α = l / r →
  α = 2 := by
sorry

end sector_central_angle_l3546_354698


namespace smallest_n_for_probability_threshold_l3546_354600

theorem smallest_n_for_probability_threshold (n : ℕ) : 
  (∀ k, k < n → 1 / (k * (k + 1)) ≥ 1 / 2010) ∧
  1 / (n * (n + 1)) < 1 / 2010 →
  n = 45 := by
  sorry

end smallest_n_for_probability_threshold_l3546_354600


namespace B_fair_share_l3546_354691

-- Define the total rent
def total_rent : ℕ := 841

-- Define the number of horses and months for each person
def horses_A : ℕ := 12
def months_A : ℕ := 8
def horses_B : ℕ := 16
def months_B : ℕ := 9
def horses_C : ℕ := 18
def months_C : ℕ := 6

-- Calculate the total horse-months
def total_horse_months : ℕ := horses_A * months_A + horses_B * months_B + horses_C * months_C

-- Calculate B's horse-months
def B_horse_months : ℕ := horses_B * months_B

-- Theorem: B's fair share of the rent is 348
theorem B_fair_share : 
  (total_rent : ℚ) * B_horse_months / total_horse_months = 348 := by
  sorry

end B_fair_share_l3546_354691


namespace min_cubes_is_four_l3546_354689

/-- Represents a cube with two protruding snaps on opposite sides and four receptacle holes --/
structure Cube where
  snaps : Fin 2
  holes : Fin 4

/-- Represents an assembly of cubes --/
def Assembly := List Cube

/-- Checks if an assembly has only receptacle holes visible --/
def Assembly.onlyHolesVisible (a : Assembly) : Prop :=
  sorry

/-- The minimum number of cubes required for a valid assembly --/
def minCubesForValidAssembly : ℕ :=
  sorry

/-- Theorem stating that the minimum number of cubes for a valid assembly is 4 --/
theorem min_cubes_is_four :
  minCubesForValidAssembly = 4 :=
sorry

end min_cubes_is_four_l3546_354689


namespace teresa_social_studies_score_l3546_354621

/-- Teresa's exam scores -/
structure ExamScores where
  science : ℕ
  music : ℕ
  physics : ℕ
  social_studies : ℕ
  total : ℕ

/-- Theorem: Given Teresa's exam scores satisfying certain conditions, her social studies score is 85 -/
theorem teresa_social_studies_score (scores : ExamScores) 
  (h1 : scores.science = 70)
  (h2 : scores.music = 80)
  (h3 : scores.physics = scores.music / 2)
  (h4 : scores.total = 275)
  (h5 : scores.total = scores.science + scores.music + scores.physics + scores.social_studies) :
  scores.social_studies = 85 := by
    sorry

#check teresa_social_studies_score

end teresa_social_studies_score_l3546_354621


namespace largest_value_in_special_sequence_l3546_354651

/-- A sequence of 8 increasing real numbers -/
def IncreasingSequence (a : Fin 8 → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j

/-- Checks if a sequence of 4 numbers is an arithmetic progression with a given common difference -/
def IsArithmeticProgression (a : Fin 4 → ℝ) (d : ℝ) : Prop :=
  ∀ i : Fin 3, a (i + 1) - a i = d

/-- Checks if a sequence of 4 numbers is a geometric progression -/
def IsGeometricProgression (a : Fin 4 → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Fin 3, a (i + 1) / a i = r

/-- The main theorem -/
theorem largest_value_in_special_sequence (a : Fin 8 → ℝ)
  (h_increasing : IncreasingSequence a)
  (h_arithmetic1 : ∃ i : Fin 5, IsArithmeticProgression (fun j => a (i + j)) 4)
  (h_arithmetic2 : ∃ i : Fin 5, IsArithmeticProgression (fun j => a (i + j)) 36)
  (h_geometric : ∃ i : Fin 5, IsGeometricProgression (fun j => a (i + j))) :
  a 7 = 126 ∨ a 7 = 6 :=
sorry

end largest_value_in_special_sequence_l3546_354651


namespace max_m_value_l3546_354696

/-- Given m > 0 and the inequality holds for all x > 0, the maximum value of m is e^2 -/
theorem max_m_value (m : ℝ) (hm : m > 0) 
  (h : ∀ x > 0, m * x * Real.log x - (x + m) * Real.exp ((x - m) / m) ≤ 0) : 
  m ≤ Real.exp 2 ∧ ∃ m₀ > 0, ∀ ε > 0, ∃ x > 0, 
    (Real.exp 2 - ε) * x * Real.log x - (x + (Real.exp 2 - ε)) * Real.exp ((x - (Real.exp 2 - ε)) / (Real.exp 2 - ε)) > 0 :=
sorry

end max_m_value_l3546_354696
