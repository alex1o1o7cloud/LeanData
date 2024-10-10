import Mathlib

namespace parallel_line_with_chord_sum_exists_l3036_303664

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line in a plane
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Theorem statement
theorem parallel_line_with_chord_sum_exists 
  (S₁ S₂ : Circle) (l : Line) (a : ℝ) (h : a > 0) :
  ∃ (l' : Line),
    (∀ (p : ℝ × ℝ), l.point.1 + p.1 * l.direction.1 = l'.point.1 + p.1 * l'.direction.1 ∧
                     l.point.2 + p.2 * l.direction.2 = l'.point.2 + p.2 * l'.direction.2) ∧
    ∃ (A B C D : ℝ × ℝ),
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = ((A.1 - S₁.center.1)^2 + (A.2 - S₁.center.2)^2 - S₁.radius^2) ∧
      (C.1 - D.1)^2 + (C.2 - D.2)^2 = ((C.1 - S₂.center.1)^2 + (C.2 - S₂.center.2)^2 - S₂.radius^2) ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 = a^2 :=
by
  sorry


end parallel_line_with_chord_sum_exists_l3036_303664


namespace group_size_calculation_l3036_303624

theorem group_size_calculation (n : ℕ) : 
  (15 * n + 35) / (n + 1) = 17 → n = 9 := by
  sorry

end group_size_calculation_l3036_303624


namespace calculate_principal_l3036_303616

/-- Given simple interest, time, and rate, calculate the principal amount -/
theorem calculate_principal (simple_interest : ℝ) (time : ℝ) (rate : ℝ) :
  simple_interest = 140 ∧ time = 2 ∧ rate = 17.5 →
  (simple_interest / (rate * time / 100) : ℝ) = 400 := by
  sorry

end calculate_principal_l3036_303616


namespace abs_sum_inequality_l3036_303695

theorem abs_sum_inequality (x : ℝ) : |x - 2| + |x + 3| < 7 ↔ -6 < x ∧ x < 3 := by
  sorry

end abs_sum_inequality_l3036_303695


namespace two_mixers_one_tv_cost_is_7000_l3036_303660

/-- The cost of a mixer in Rupees -/
def mixer_cost : ℕ := sorry

/-- The cost of a TV in Rupees -/
def tv_cost : ℕ := 4200

/-- The total cost of two mixers and one TV in Rupees -/
def two_mixers_one_tv_cost : ℕ := 2 * mixer_cost + tv_cost

theorem two_mixers_one_tv_cost_is_7000 :
  (two_mixers_one_tv_cost = 7000) ∧ (2 * tv_cost + mixer_cost = 9800) :=
sorry

end two_mixers_one_tv_cost_is_7000_l3036_303660


namespace car_journey_time_l3036_303634

/-- Calculates the total time for a car journey with two segments and a stop -/
theorem car_journey_time (distance1 : ℝ) (speed1 : ℝ) (stop_time : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  distance1 = 150 ∧ speed1 = 50 ∧ stop_time = 0.5 ∧ distance2 = 200 ∧ speed2 = 75 →
  distance1 / speed1 + stop_time + distance2 / speed2 = 6.17 := by
  sorry

#eval (150 / 50 + 0.5 + 200 / 75 : Float)

end car_journey_time_l3036_303634


namespace total_students_count_l3036_303625

/-- The total number of students in a primary school height survey. -/
def total_students : ℕ := 621

/-- The number of students with heights not exceeding 130 cm. -/
def students_under_130 : ℕ := 99

/-- The average height of students not exceeding 130 cm, in cm. -/
def avg_height_under_130 : ℝ := 122

/-- The number of students with heights not less than 160 cm. -/
def students_over_160 : ℕ := 72

/-- The average height of students not less than 160 cm, in cm. -/
def avg_height_over_160 : ℝ := 163

/-- The average height of students exceeding 130 cm, in cm. -/
def avg_height_130_to_160 : ℝ := 155

/-- The average height of students below 160 cm, in cm. -/
def avg_height_under_160 : ℝ := 148

/-- Theorem stating that given the conditions, the total number of students is 621. -/
theorem total_students_count : total_students = students_under_130 + students_over_160 + 
  (total_students - students_under_130 - students_over_160) :=
by sorry

end total_students_count_l3036_303625


namespace exists_noninteger_zero_point_l3036_303601

/-- Definition of the polynomial p(x,y) -/
def p (b : Fin 12 → ℝ) (x y : ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4

/-- The theorem stating the existence of a non-integer point (r,s) where p(r,s) = 0 -/
theorem exists_noninteger_zero_point :
  ∃ (r s : ℝ), ¬(∃ m n : ℤ, (r : ℝ) = m ∧ (s : ℝ) = n) ∧
    ∀ (b : Fin 12 → ℝ), 
      p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧ 
      p b 0 1 = 0 ∧ p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧ 
      p b 1 (-1) = 0 ∧ p b 2 2 = 0 ∧ p b (-1) (-1) = 0 →
      p b r s = 0 :=
sorry

end exists_noninteger_zero_point_l3036_303601


namespace jason_seashells_l3036_303693

theorem jason_seashells (initial_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : initial_seashells = 49) (h2 : given_seashells = 13) :
  initial_seashells - given_seashells = 36 := by
  sorry

#check jason_seashells

end jason_seashells_l3036_303693


namespace total_triangles_is_nine_l3036_303607

/-- Represents a triangular grid with a specific number of rows and triangles per row. -/
structure TriangularGrid where
  rows : Nat
  triangles_per_row : Nat → Nat
  row_count_correct : rows = 3
  top_row_correct : triangles_per_row 0 = 3
  second_row_correct : triangles_per_row 1 = 2
  bottom_row_correct : triangles_per_row 2 = 1

/-- Calculates the total number of triangles in the grid, including larger triangles formed by combining smaller ones. -/
def totalTriangles (grid : TriangularGrid) : Nat :=
  sorry

/-- Theorem stating that the total number of triangles in the specified triangular grid is 9. -/
theorem total_triangles_is_nine (grid : TriangularGrid) : totalTriangles grid = 9 := by
  sorry

end total_triangles_is_nine_l3036_303607


namespace max_silver_tokens_l3036_303680

/-- Represents the state of tokens --/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth --/
structure ExchangeBooth where
  redIn : ℕ
  blueIn : ℕ
  redOut : ℕ
  blueOut : ℕ
  silverOut : ℕ

/-- Defines if an exchange is possible given a token state and an exchange booth --/
def canExchange (state : TokenState) (booth : ExchangeBooth) : Prop :=
  state.red ≥ booth.redIn ∧ state.blue ≥ booth.blueIn

/-- Defines the result of an exchange --/
def exchangeResult (state : TokenState) (booth : ExchangeBooth) : TokenState :=
  { red := state.red - booth.redIn + booth.redOut,
    blue := state.blue - booth.blueIn + booth.blueOut,
    silver := state.silver + booth.silverOut }

/-- Defines if a state is final (no more exchanges possible) --/
def isFinalState (state : TokenState) (booths : List ExchangeBooth) : Prop :=
  ∀ booth ∈ booths, ¬(canExchange state booth)

/-- The main theorem --/
theorem max_silver_tokens : 
  ∃ (finalState : TokenState),
    let initialState : TokenState := { red := 100, blue := 50, silver := 0 }
    let booth1 : ExchangeBooth := { redIn := 4, blueIn := 0, redOut := 0, blueOut := 3, silverOut := 1 }
    let booth2 : ExchangeBooth := { redIn := 0, blueIn := 2, redOut := 1, blueOut := 0, silverOut := 1 }
    let booths : List ExchangeBooth := [booth1, booth2]
    (isFinalState finalState booths) ∧
    (finalState.silver = 143) ∧
    (∀ (otherFinalState : TokenState),
      (isFinalState otherFinalState booths) →
      (otherFinalState.silver ≤ finalState.silver)) := by
  sorry


end max_silver_tokens_l3036_303680


namespace prob_sum_greater_than_9_l3036_303647

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (sum > 9) -/
def favorableOutcomes : ℕ := 6

/-- The probability of rolling a sum greater than 9 with two dice -/
def probSumGreaterThan9 : ℚ := favorableOutcomes / totalOutcomes

theorem prob_sum_greater_than_9 : probSumGreaterThan9 = 1 / 6 := by
  sorry

end prob_sum_greater_than_9_l3036_303647


namespace banana_pile_count_l3036_303643

/-- The total number of bananas in a pile after adding more bananas -/
def total_bananas (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: Given 2 initial bananas and 7 added bananas, the total is 9 -/
theorem banana_pile_count : total_bananas 2 7 = 9 := by
  sorry

end banana_pile_count_l3036_303643


namespace guitar_strings_problem_l3036_303687

theorem guitar_strings_problem (total_strings : ℕ) 
  (num_basses : ℕ) (strings_per_bass : ℕ) 
  (strings_per_normal_guitar : ℕ) (strings_difference : ℕ) :
  let num_normal_guitars := 2 * num_basses
  let strings_for_basses := num_basses * strings_per_bass
  let strings_for_normal_guitars := num_normal_guitars * strings_per_normal_guitar
  let remaining_strings := total_strings - strings_for_basses - strings_for_normal_guitars
  let strings_per_fewer_guitar := strings_per_normal_guitar - strings_difference
  total_strings = 72 ∧ 
  num_basses = 3 ∧ 
  strings_per_bass = 4 ∧ 
  strings_per_normal_guitar = 6 ∧ 
  strings_difference = 3 →
  strings_per_fewer_guitar = 3 :=
by sorry

end guitar_strings_problem_l3036_303687


namespace triplet_sum_not_two_l3036_303661

theorem triplet_sum_not_two : ∃! (x y z : ℝ), 
  ((x = 2.2 ∧ y = -3.2 ∧ z = 2.0) ∨
   (x = 3/4 ∧ y = 1/2 ∧ z = 3/4) ∨
   (x = 4 ∧ y = -6 ∧ z = 4) ∨
   (x = 0.4 ∧ y = 0.5 ∧ z = 1.1) ∨
   (x = 2/3 ∧ y = 1/3 ∧ z = 1)) ∧
  x + y + z ≠ 2 :=
by sorry

end triplet_sum_not_two_l3036_303661


namespace g_behavior_at_infinity_l3036_303641

def g (x : ℝ) : ℝ := -3 * x^4 + 15 * x^2 - 10

theorem g_behavior_at_infinity :
  (∀ ε > 0, ∃ N > 0, ∀ x : ℝ, x > N → g x < -ε) ∧
  (∀ ε > 0, ∃ N > 0, ∀ x : ℝ, x < -N → g x < -ε) :=
sorry

end g_behavior_at_infinity_l3036_303641


namespace revenue_equals_scientific_notation_l3036_303608

/-- Represents the total revenue in yuan -/
def total_revenue : ℝ := 998.64e9

/-- Represents the scientific notation of the total revenue -/
def scientific_notation : ℝ := 9.9864e11

/-- Theorem stating that the total revenue is equal to its scientific notation representation -/
theorem revenue_equals_scientific_notation : total_revenue = scientific_notation := by
  sorry

end revenue_equals_scientific_notation_l3036_303608


namespace abc_inequality_l3036_303632

theorem abc_inequality (a b c : ℝ) (ha : |a| < 1) (hb : |b| < 1) (hc : |c| < 1) :
  a * b * c + 2 > a + b + c := by
  sorry

end abc_inequality_l3036_303632


namespace minimum_distance_theorem_l3036_303604

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 2

def line_l (x y : ℝ) : Prop := x + 2 * y - 2 * Real.log 2 - 6 = 0

def M (px py qx qy : ℝ) : ℝ := (px - qx)^2 + (py - qy)^2

theorem minimum_distance_theorem (px py qx qy : ℝ) 
  (h1 : f px = py) 
  (h2 : line_l qx qy) : 
  (∃ (min_M : ℝ), ∀ (px' py' qx' qy' : ℝ), 
    f px' = py' → line_l qx' qy' → 
    M px' py' qx' qy' ≥ min_M ∧ 
    min_M = 16/5 ∧
    (M px py qx qy = min_M → qx = 14/5)) := by sorry

end minimum_distance_theorem_l3036_303604


namespace sandwich_counts_l3036_303620

def is_valid_sandwich_count (s : ℕ) : Prop :=
  ∃ (c : ℕ), 
    s + c = 7 ∧ 
    (100 * s + 75 * c) % 100 = 0

theorem sandwich_counts : 
  ∀ s : ℕ, is_valid_sandwich_count s ↔ (s = 3 ∨ s = 7) :=
by sorry

end sandwich_counts_l3036_303620


namespace smallest_valid_configuration_l3036_303623

/-- Represents a bench configuration at a concert --/
structure BenchConfiguration where
  M : ℕ  -- Number of bench sections
  adultsPerBench : ℕ  -- Number of adults per bench
  childrenPerBench : ℕ  -- Number of children per bench

/-- Checks if a given bench configuration is valid --/
def isValidConfiguration (config : BenchConfiguration) : Prop :=
  ∃ (adults children : ℕ),
    adults + children = config.M * config.adultsPerBench ∧
    children = 2 * adults ∧
    children ≤ config.M * config.childrenPerBench

/-- The theorem to be proved --/
theorem smallest_valid_configuration :
  ∃ (config : BenchConfiguration),
    config.M = 6 ∧
    config.adultsPerBench = 8 ∧
    config.childrenPerBench = 12 ∧
    isValidConfiguration config ∧
    (∀ (otherConfig : BenchConfiguration),
      otherConfig.adultsPerBench = 8 →
      otherConfig.childrenPerBench = 12 →
      isValidConfiguration otherConfig →
      otherConfig.M ≥ config.M) :=
  sorry

end smallest_valid_configuration_l3036_303623


namespace set_relations_l3036_303662

variable {α : Type*}
variable (I A B : Set α)

theorem set_relations (h : A ∪ B = I) :
  (Aᶜ ∩ Bᶜ = ∅) ∧ (B ⊇ Aᶜ) := by
  sorry

end set_relations_l3036_303662


namespace stratified_sampling_female_count_l3036_303657

theorem stratified_sampling_female_count
  (total_students : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = male_students + female_students)
  (h2 : total_students = 49)
  (h3 : male_students = 28)
  (h4 : female_students = 21)
  (h5 : sample_size = 14) :
  (sample_size : ℚ) / total_students * female_students = 6 :=
by sorry

end stratified_sampling_female_count_l3036_303657


namespace area_of_closed_figure_l3036_303658

/-- The area of the closed figure bounded by y = 1/2, y = 2, y = 1/x, and the y-axis is 2ln(2) -/
theorem area_of_closed_figure : 
  let lower_bound : ℝ := 1/2
  let upper_bound : ℝ := 2
  let curve (x : ℝ) : ℝ := 1/x
  ∫ y in lower_bound..upper_bound, (1/y) = 2 * Real.log 2 := by
  sorry

end area_of_closed_figure_l3036_303658


namespace divisibility_theorem_l3036_303675

theorem divisibility_theorem (N : ℕ) (h : N > 1) :
  ∃ k : ℤ, (N^2)^2014 - (N^11)^106 = k * (N^6 + N^3 + 1) := by sorry

end divisibility_theorem_l3036_303675


namespace circle_problem_l3036_303602

-- Define the circles and points
def largeCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 100}
def smallCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 36}
def P : ℝ × ℝ := (6, 8)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- State the theorem
theorem circle_problem (k : ℝ) 
  (h1 : P ∈ largeCircle) 
  (h2 : S k ∈ smallCircle) 
  (h3 : (10 : ℝ) - (6 : ℝ) = 4) : 
  k = 6 := by sorry

end circle_problem_l3036_303602


namespace socks_theorem_l3036_303654

def socks_problem (week1 week2 week3 week4 total : ℕ) : Prop :=
  week1 = 12 ∧
  week2 = week1 + 4 ∧
  week3 = (week1 + week2) / 2 ∧
  total = 57 ∧
  total = week1 + week2 + week3 + week4 ∧
  week3 - week4 = 1

theorem socks_theorem : ∃ week1 week2 week3 week4 total : ℕ,
  socks_problem week1 week2 week3 week4 total := by
  sorry

end socks_theorem_l3036_303654


namespace intersection_A_B_l3036_303653

def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end intersection_A_B_l3036_303653


namespace xyz_value_l3036_303626

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + y * z + z * x) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 14/3 := by
sorry

end xyz_value_l3036_303626


namespace imaginary_part_of_z_l3036_303686

theorem imaginary_part_of_z (z : ℂ) (h : (z - Complex.I) / (z - 2) = Complex.I) :
  z.im = -1/2 := by sorry

end imaginary_part_of_z_l3036_303686


namespace chocolate_milk_ounces_l3036_303628

/-- The number of ounces of milk in each glass of chocolate milk. -/
def milk_per_glass : ℚ := 13/2

/-- The number of ounces of chocolate syrup in each glass of chocolate milk. -/
def syrup_per_glass : ℚ := 3/2

/-- The total number of ounces in each glass of chocolate milk. -/
def total_per_glass : ℚ := milk_per_glass + syrup_per_glass

/-- Theorem stating that each glass of chocolate milk contains 8 ounces. -/
theorem chocolate_milk_ounces : total_per_glass = 8 := by
  sorry

end chocolate_milk_ounces_l3036_303628


namespace arithmetic_mean_problem_l3036_303619

theorem arithmetic_mean_problem (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 4 → x = 28 := by
  sorry

end arithmetic_mean_problem_l3036_303619


namespace roots_polynomial_sum_l3036_303671

theorem roots_polynomial_sum (α β : ℝ) : 
  α^2 - 3*α - 4 = 0 ∧ β^2 - 3*β - 4 = 0 → 3*α^4 + 8*β^3 = 876 := by
  sorry

end roots_polynomial_sum_l3036_303671


namespace expansion_coefficient_l3036_303613

/-- The coefficient of x^(3/2) in the expansion of (√x - a/x)^6 -/
def coefficient (a : ℝ) : ℝ := 6 * (-a)

theorem expansion_coefficient (a : ℝ) : coefficient a = 30 → a = -5 := by
  sorry

end expansion_coefficient_l3036_303613


namespace geometric_sequence_proof_minimum_years_proof_l3036_303650

def initial_capital : ℝ := 50
def growth_rate : ℝ := 0.5
def payment (t : ℝ) : ℝ := t

def remaining_capital (n : ℕ) (t : ℝ) : ℝ :=
  if n = 0 then initial_capital
  else (1 + growth_rate) * remaining_capital (n - 1) t - payment t

theorem geometric_sequence_proof (t : ℝ) (h : 0 < t ∧ t < 2500) :
  ∀ n : ℕ, (remaining_capital (n + 1) t - 2 * t) / (remaining_capital n t - 2 * t) = 3 / 2 :=
sorry

theorem minimum_years_proof :
  let t := 1500
  (∃ m : ℕ, remaining_capital m t > 21000) ∧
  (∀ k : ℕ, k < 6 → remaining_capital k t ≤ 21000) :=
sorry

end geometric_sequence_proof_minimum_years_proof_l3036_303650


namespace sum_ends_with_1379_l3036_303640

theorem sum_ends_with_1379 (S : Finset ℕ) (h1 : S.card = 10000) 
  (h2 : ∀ n ∈ S, Odd n ∧ ¬(5 ∣ n)) : 
  ∃ T ⊆ S, (T.sum id) % 10000 = 1379 := by
sorry

end sum_ends_with_1379_l3036_303640


namespace estimate_larger_than_actual_l3036_303636

theorem estimate_larger_than_actual (x y z : ℝ) 
  (h1 : x > y) (h2 : y > 0) (h3 : z > 0) : 
  (x + 2*z) - (y - 2*z) > x - y := by
  sorry

end estimate_larger_than_actual_l3036_303636


namespace teacher_student_arrangement_l3036_303666

theorem teacher_student_arrangement (n : ℕ) (m : ℕ) :
  n = 1 ∧ m = 6 →
  (n + m - 2) * (m.factorial) = 3600 :=
by sorry

end teacher_student_arrangement_l3036_303666


namespace intersection_volume_is_constant_l3036_303638

def cube_side_length : ℝ := 6

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def is_inside_cube (p : Point3D) : Prop :=
  0 < p.x ∧ p.x < cube_side_length ∧
  0 < p.y ∧ p.y < cube_side_length ∧
  0 < p.z ∧ p.z < cube_side_length

def intersection_volume (p : Point3D) : ℝ :=
  cube_side_length ^ 3 - cube_side_length ^ 2

theorem intersection_volume_is_constant (p : Point3D) (h : is_inside_cube p) :
  intersection_volume p = 180 := by sorry

end intersection_volume_is_constant_l3036_303638


namespace fresh_grapes_water_content_l3036_303672

/-- Percentage of water in dried grapes -/
def dried_water_percentage : ℝ := 20

/-- Weight of fresh grapes in kg -/
def fresh_weight : ℝ := 40

/-- Weight of dried grapes in kg -/
def dried_weight : ℝ := 10

/-- Percentage of water in fresh grapes -/
def fresh_water_percentage : ℝ := 80

theorem fresh_grapes_water_content :
  (1 - fresh_water_percentage / 100) * fresh_weight = (1 - dried_water_percentage / 100) * dried_weight :=
sorry

end fresh_grapes_water_content_l3036_303672


namespace power_less_than_threshold_l3036_303673

theorem power_less_than_threshold : ∃ (n1 n2 n3 : ℕ+),
  (0.99 : ℝ) ^ (n1 : ℝ) < 0.000001 ∧
  (0.999 : ℝ) ^ (n2 : ℝ) < 0.000001 ∧
  (0.999999 : ℝ) ^ (n3 : ℝ) < 0.000001 := by
  sorry

end power_less_than_threshold_l3036_303673


namespace power_difference_divisibility_l3036_303667

theorem power_difference_divisibility (N : ℕ+) :
  ∃ (r s : ℕ), r ≠ s ∧ ∀ (A : ℤ), (N : ℤ) ∣ (A^r - A^s) := by
  sorry

end power_difference_divisibility_l3036_303667


namespace max_value_quadratic_l3036_303691

theorem max_value_quadratic (q : ℝ) : -3 * q^2 + 18 * q + 5 ≤ 32 ∧ ∃ q₀ : ℝ, -3 * q₀^2 + 18 * q₀ + 5 = 32 := by
  sorry

end max_value_quadratic_l3036_303691


namespace pond_volume_pond_volume_proof_l3036_303646

/-- The volume of a rectangular prism with dimensions 20 m, 10 m, and 5 m is 1000 cubic meters. -/
theorem pond_volume : ℝ → Prop :=
  fun volume =>
    let length : ℝ := 20
    let width : ℝ := 10
    let depth : ℝ := 5
    volume = length * width * depth ∧ volume = 1000

/-- Proof of the pond volume theorem -/
theorem pond_volume_proof : ∃ volume : ℝ, pond_volume volume := by
  sorry

end pond_volume_pond_volume_proof_l3036_303646


namespace sector_to_cone_l3036_303644

/-- Represents a cone formed from a circular sector -/
structure SectorCone where
  sector_radius : ℝ
  sector_angle : ℝ
  base_radius : ℝ
  slant_height : ℝ

/-- Theorem: A 270° sector of a circle with radius 12 forms a cone with base radius 9 and slant height 12 -/
theorem sector_to_cone :
  ∀ (cone : SectorCone),
    cone.sector_radius = 12 ∧
    cone.sector_angle = 270 ∧
    cone.slant_height = cone.sector_radius →
    cone.base_radius = 9 ∧
    cone.slant_height = 12 := by
  sorry


end sector_to_cone_l3036_303644


namespace initial_number_proof_l3036_303665

theorem initial_number_proof (x : ℝ) : 
  ((5 * x - 20) / 2 - 100 = 4) → x = 45.6 := by
sorry

end initial_number_proof_l3036_303665


namespace smallest_n_for_non_prime_2n_plus_1_l3036_303685

theorem smallest_n_for_non_prime_2n_plus_1 :
  ∃ n : ℕ+, (∀ k < n, Nat.Prime (2 * k + 1)) ∧ ¬Nat.Prime (2 * n + 1) ∧ n = 4 := by
  sorry

end smallest_n_for_non_prime_2n_plus_1_l3036_303685


namespace parallel_line_slope_l3036_303694

theorem parallel_line_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let given_line := {(x, y) : ℝ × ℝ | a * x - b * y = c}
  let slope := a / b
  ∀ m : ℝ, (∃ k : ℝ, ∀ x y : ℝ, y = m * x + k ↔ (x, y) ∈ given_line) → m = slope :=
by sorry

end parallel_line_slope_l3036_303694


namespace min_sum_with_conditions_l3036_303681

theorem min_sum_with_conditions (a b : ℕ+) 
  (h1 : ¬ 5 ∣ a.val)
  (h2 : ¬ 5 ∣ b.val)
  (h3 : (5 : ℕ)^5 ∣ a.val^5 + b.val^5) :
  ∀ (x y : ℕ+), 
    (¬ 5 ∣ x.val) → 
    (¬ 5 ∣ y.val) → 
    ((5 : ℕ)^5 ∣ x.val^5 + y.val^5) → 
    (a.val + b.val ≤ x.val + y.val) :=
by sorry

end min_sum_with_conditions_l3036_303681


namespace misread_division_sum_l3036_303689

theorem misread_division_sum (D : ℕ) (h : D = 56 * 25 + 25) :
  ∃ (q r : ℕ), D = 65 * q + r ∧ r < 65 ∧ q + r = 81 := by
  sorry

end misread_division_sum_l3036_303689


namespace quadratic_equation_solution_l3036_303635

/-- Given a quadratic equation (m-2)x^2 + 3x + m^2 - 4 = 0 where x = 0 is a solution, prove that m = -2 -/
theorem quadratic_equation_solution (m : ℝ) : 
  ((m - 2) * 0^2 + 3 * 0 + m^2 - 4 = 0) → m = -2 :=
by sorry

end quadratic_equation_solution_l3036_303635


namespace special_circle_distances_l3036_303668

/-- A circle with specific properties and a point on its circumference -/
structure SpecialCircle where
  r : ℕ
  u : ℕ
  v : ℕ
  p : ℕ
  q : ℕ
  m : ℕ
  n : ℕ
  h_r_odd : Odd r
  h_circle_eq : u^2 + v^2 = r^2
  h_u_prime_power : u = p^m
  h_v_prime_power : v = q^n
  h_p_prime : Nat.Prime p
  h_q_prime : Nat.Prime q
  h_u_gt_v : u > v

/-- The theorem to be proved -/
theorem special_circle_distances (c : SpecialCircle) :
  let A : ℝ × ℝ := (c.r, 0)
  let B : ℝ × ℝ := (-c.r, 0)
  let C : ℝ × ℝ := (0, -c.r)
  let D : ℝ × ℝ := (0, c.r)
  let P : ℝ × ℝ := (c.u, c.v)
  let M : ℝ × ℝ := (c.u, 0)
  let N : ℝ × ℝ := (0, c.v)
  |A.1 - M.1| = 1 ∧
  |B.1 - M.1| = 9 ∧
  |C.2 - N.2| = 8 ∧
  |D.2 - N.2| = 2 :=
by sorry

end special_circle_distances_l3036_303668


namespace sum_longest_altitudes_is_14_l3036_303605

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10
  right_angle : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the triangle -/
def sum_longest_altitudes (t : RightTriangle) : ℝ := t.a + t.b

/-- Theorem: The sum of the lengths of the two longest altitudes in a triangle 
    with sides 6, 8, and 10 is 14 -/
theorem sum_longest_altitudes_is_14 (t : RightTriangle) : 
  sum_longest_altitudes t = 14 := by
  sorry

end sum_longest_altitudes_is_14_l3036_303605


namespace solution_properties_l3036_303698

def system_of_equations (t x y : ℝ) : Prop :=
  (4*t^2 + t + 4)*x + (5*t + 1)*y = 4*t^2 - t - 3 ∧
  (t + 2)*x + 2*y = t

theorem solution_properties :
  ∀ t : ℝ,
  (∀ x y : ℝ, system_of_equations t x y →
    (t < -1 → x < 0 ∧ y < 0) ∧
    (-1 < t ∧ t < 1 ∧ t ≠ 0 → x = (t+1)/(t-1) ∧ y = (2*t+1)/(t-1)) ∧
    (t = 1 → ∀ k : ℝ, ∃ x y : ℝ, system_of_equations t x y) ∧
    (t = 2 → ¬∃ x y : ℝ, system_of_equations t x y) ∧
    (t > 2 → x > 0 ∧ y > 0)) :=
by sorry

end solution_properties_l3036_303698


namespace s_equals_2012_l3036_303677

/-- S(n, k) is the number of coefficients in the expansion of (x+1)^n that are not divisible by k -/
def S (n k : ℕ) : ℕ := sorry

/-- Theorem stating that S(2012^2011, 2011) equals 2012 -/
theorem s_equals_2012 : S (2012^2011) 2011 = 2012 := by sorry

end s_equals_2012_l3036_303677


namespace age_problem_l3036_303655

/-- The age problem -/
theorem age_problem (a b : ℝ) : 
  b = 3 * a - 20 →  -- Bob's age is 20 years less than three times Alice's age
  a + b = 70 →      -- The sum of their ages is 70
  b = 47.5 :=       -- Bob's age is 47.5
by sorry

end age_problem_l3036_303655


namespace sofia_running_time_l3036_303629

/-- The time Sofia takes to complete 8 laps on a track with given conditions -/
theorem sofia_running_time (laps : ℕ) (track_length : ℝ) (first_half_speed : ℝ) (second_half_speed : ℝ)
  (h1 : laps = 8)
  (h2 : track_length = 300)
  (h3 : first_half_speed = 5)
  (h4 : second_half_speed = 6) :
  let time_per_lap := track_length / (2 * first_half_speed) + track_length / (2 * second_half_speed)
  let total_time := laps * time_per_lap
  total_time = 440 := by sorry

#eval (7 * 60 + 20 : ℕ) -- Evaluates to 440, confirming 7 minutes and 20 seconds

end sofia_running_time_l3036_303629


namespace arithmetic_sequence_sum_l3036_303642

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 5 + a 8 + a 11 = 48 →
  a 6 + a 7 = 24 := by
sorry

end arithmetic_sequence_sum_l3036_303642


namespace triangle_properties_l3036_303603

/-- An acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- The theorem stating the properties of the specific triangle -/
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : t.a = 2 * t.b * Real.sin t.A)
  (h2 : t.a = 3 * Real.sqrt 3)
  (h3 : t.c = 5) :
  t.B = π/6 ∧ t.b = Real.sqrt 7 := by
  sorry

end triangle_properties_l3036_303603


namespace x_to_y_equals_nine_l3036_303627

theorem x_to_y_equals_nine (x y : ℝ) : y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 2 → x^y = 9 := by
  sorry

end x_to_y_equals_nine_l3036_303627


namespace negative_fraction_comparison_l3036_303645

theorem negative_fraction_comparison : -5/6 < -4/5 := by
  sorry

end negative_fraction_comparison_l3036_303645


namespace tenth_odd_multiple_of_5_l3036_303676

/-- The nth odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Predicate for a number being odd and a multiple of 5 -/
def isOddMultipleOf5 (k : ℕ) : Prop := k % 2 = 1 ∧ k % 5 = 0

theorem tenth_odd_multiple_of_5 : 
  (∃ (k : ℕ), k > 0 ∧ isOddMultipleOf5 k ∧ 
    (∃ (count : ℕ), count = 10 ∧ 
      (∀ (j : ℕ), j > 0 ∧ j < k → isOddMultipleOf5 j → 
        (∃ (m : ℕ), m < count ∧ nthOddMultipleOf5 m = j)))) → 
  nthOddMultipleOf5 10 = 95 :=
sorry

end tenth_odd_multiple_of_5_l3036_303676


namespace four_different_suits_count_l3036_303659

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents a suit in a deck of cards -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- Number of suits in a standard deck -/
def num_suits : Nat := 4

/-- Number of cards in each suit -/
def cards_per_suit : Nat := 13

/-- 
Theorem: The number of ways to choose 4 cards from a standard deck of 52 cards, 
where all four cards must be of different suits and the order doesn't matter, 
is equal to 28561.
-/
theorem four_different_suits_count (d : Deck) : 
  (cards_per_suit ^ num_suits) = 28561 := by
  sorry

end four_different_suits_count_l3036_303659


namespace wind_speed_calculation_l3036_303617

/-- Given a jet's flight conditions, prove the wind speed is 50 mph -/
theorem wind_speed_calculation (j w : ℝ) 
  (h1 : 2000 = (j + w) * 4)   -- Equation for flight with tailwind
  (h2 : 2000 = (j - w) * 5)   -- Equation for return flight against wind
  : w = 50 := by
  sorry

end wind_speed_calculation_l3036_303617


namespace circle_line_intersection_l3036_303611

/-- Circle C: x^2 + y^2 - 2x + 2y - 4 = 0 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 4 = 0

/-- Line l: y = x + b with slope 1 -/
def Line (x y b : ℝ) : Prop := y = x + b

/-- Intersection points of Circle and Line -/
def Intersection (b : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  x₁ ≠ x₂ ∧ Circle x₁ y₁ ∧ Circle x₂ y₂ ∧ Line x₁ y₁ b ∧ Line x₂ y₂ b

/-- The circle with diameter AB passes through the origin -/
def CircleThroughOrigin (b : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ,
  Circle x₁ y₁ ∧ Circle x₂ y₂ ∧ Line x₁ y₁ b ∧ Line x₂ y₂ b ∧
  x₁*x₂ + y₁*y₂ = 0

theorem circle_line_intersection :
  (∀ b : ℝ, Intersection b ↔ -3-3*Real.sqrt 2 < b ∧ b < -3+3*Real.sqrt 2) ∧
  (∃! b₁ b₂ : ℝ, b₁ ≠ b₂ ∧ CircleThroughOrigin b₁ ∧ CircleThroughOrigin b₂ ∧
    b₁ = 1 ∧ b₂ = -4) :=
sorry

end circle_line_intersection_l3036_303611


namespace age_difference_robert_elizabeth_l3036_303669

theorem age_difference_robert_elizabeth : 
  ∀ (robert_age patrick_age elizabeth_age : ℕ),
  robert_age = 28 →
  patrick_age = robert_age / 2 →
  elizabeth_age = patrick_age - 4 →
  robert_age - elizabeth_age = 18 :=
by
  sorry

end age_difference_robert_elizabeth_l3036_303669


namespace fruit_juice_needed_correct_problem_solution_l3036_303633

/-- Represents the ratio of ingredients in a drink -/
structure DrinkRatio where
  milk : ℚ
  fruit_juice : ℚ

/-- Represents the amount of ingredients in a drink -/
structure DrinkAmount where
  milk : ℚ
  fruit_juice : ℚ

/-- Converts a ratio to normalized form where total parts sum to 1 -/
def normalize_ratio (r : DrinkRatio) : DrinkRatio :=
  let total := r.milk + r.fruit_juice
  { milk := r.milk / total, fruit_juice := r.fruit_juice / total }

/-- Calculates the amount of fruit juice needed to convert drink A to drink B -/
def fruit_juice_needed (amount_A : ℚ) (ratio_A ratio_B : DrinkRatio) : ℚ :=
  let norm_A := normalize_ratio ratio_A
  let norm_B := normalize_ratio ratio_B
  let milk_A := amount_A * norm_A.milk
  let fruit_juice_A := amount_A * norm_A.fruit_juice
  (milk_A - fruit_juice_A) / (norm_B.fruit_juice - norm_B.milk)

/-- Theorem: The amount of fruit juice needed is correct -/
theorem fruit_juice_needed_correct (amount_A : ℚ) (ratio_A ratio_B : DrinkRatio) :
  let juice_needed := fruit_juice_needed amount_A ratio_A ratio_B
  let total_amount := amount_A + juice_needed
  let final_amount := DrinkAmount.mk (amount_A * (normalize_ratio ratio_A).milk) (fruit_juice_needed amount_A ratio_A ratio_B + amount_A * (normalize_ratio ratio_A).fruit_juice)
  final_amount.milk / total_amount = (normalize_ratio ratio_B).milk ∧
  final_amount.fruit_juice / total_amount = (normalize_ratio ratio_B).fruit_juice :=
by sorry

/-- Specific problem instance -/
def drink_A : DrinkRatio := { milk := 4, fruit_juice := 3 }
def drink_B : DrinkRatio := { milk := 3, fruit_juice := 4 }

/-- Theorem: For the given problem, 14 liters of fruit juice are needed -/
theorem problem_solution : 
  fruit_juice_needed 98 drink_A drink_B = 14 :=
by sorry

end fruit_juice_needed_correct_problem_solution_l3036_303633


namespace fractional_unit_problem_l3036_303631

def fractional_unit (n : ℕ) (d : ℕ) : ℚ := 1 / d

def smallest_prime : ℕ := 2

theorem fractional_unit_problem (n d : ℕ) (h : n = 13 ∧ d = 5) :
  let x := fractional_unit n d
  x = 1/5 ∧ n * x - 3 * x = smallest_prime := by sorry

end fractional_unit_problem_l3036_303631


namespace exists_30digit_root_l3036_303651

/-- A function that checks if a number is a three-digit natural number -/
def isThreeDigitNatural (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The main theorem -/
theorem exists_30digit_root (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ) 
  (h₀ : isThreeDigitNatural a₀)
  (h₁ : isThreeDigitNatural a₁)
  (h₂ : isThreeDigitNatural a₂)
  (h₃ : isThreeDigitNatural a₃)
  (h₄ : isThreeDigitNatural a₄)
  (h₅ : isThreeDigitNatural a₅)
  (h₆ : isThreeDigitNatural a₆)
  (h₇ : isThreeDigitNatural a₇)
  (h₈ : isThreeDigitNatural a₈)
  (h₉ : isThreeDigitNatural a₉) :
  ∃ (N : ℕ) (x : ℤ), 
    (N ≥ 10^29 ∧ N < 10^30) ∧ 
    (a₉ * x^9 + a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + 
     a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀ = N) := by
  sorry

end exists_30digit_root_l3036_303651


namespace vertical_asymptote_at_neg_two_l3036_303606

/-- The function f(x) = (x^2 + 6x + 9) / (x + 2) has a vertical asymptote at x = -2 -/
theorem vertical_asymptote_at_neg_two :
  ∃ (f : ℝ → ℝ), 
    (∀ x ≠ -2, f x = (x^2 + 6*x + 9) / (x + 2)) ∧
    (∃ (L : ℝ → ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x + 2| ∧ |x + 2| < δ → |f x| > L ε)) :=
by sorry

end vertical_asymptote_at_neg_two_l3036_303606


namespace max_min_values_of_f_l3036_303690

def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = min) ∧
    max = 4 ∧ min = -5 :=
by sorry

end max_min_values_of_f_l3036_303690


namespace broken_stick_triangle_area_l3036_303699

/-- Given three sticks of length 24, if one is broken to form a right triangle with the others,
    the area of the resulting triangle is 216. -/
theorem broken_stick_triangle_area : 
  ∀ a : ℝ, 0 < a → a < 24 →
  (a^2 + 24^2 = (48 - a)^2) →
  (1/2 * a * 24 = 216) :=
by sorry

end broken_stick_triangle_area_l3036_303699


namespace min_value_quadratic_function_l3036_303679

theorem min_value_quadratic_function :
  ∀ (x y z : ℝ), 
    x^2 + 4*x*y + 3*y^2 + 2*z^2 - 8*x - 4*y + 6*z ≥ -13.5 ∧
    (x^2 + 4*x*y + 3*y^2 + 2*z^2 - 8*x - 4*y + 6*z = -13.5 ↔ x = 1 ∧ y = 3/2 ∧ z = -3/2) :=
by sorry

end min_value_quadratic_function_l3036_303679


namespace ferris_wheel_capacity_l3036_303684

theorem ferris_wheel_capacity (total_people : ℕ) (total_seats : ℕ) 
  (h1 : total_people = 16) (h2 : total_seats = 4) : 
  total_people / total_seats = 4 := by
  sorry

end ferris_wheel_capacity_l3036_303684


namespace divisible_by_eleven_l3036_303610

/-- The number formed by concatenating digits a, 7, 1, 9 in that order -/
def number (a : ℕ) : ℕ := a * 1000 + 719

/-- The alternating sum of digits used in the divisibility rule for 11 -/
def alternating_sum (a : ℕ) : ℤ := a - 7 + 1 - 9

theorem divisible_by_eleven (a : ℕ) : 
  (0 ≤ a ∧ a ≤ 9) → (number a % 11 = 0 ↔ a = 4) := by
  sorry

end divisible_by_eleven_l3036_303610


namespace hyperbola_a_value_l3036_303621

-- Define the hyperbola equation
def hyperbola_eq (x y a : ℝ) : Prop := x^2 / (a + 3) - y^2 / 3 = 1

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_a_value :
  ∃ (a : ℝ), (∀ (x y : ℝ), hyperbola_eq x y a) ∧ 
  (eccentricity = 2) → a = -2 :=
sorry

end hyperbola_a_value_l3036_303621


namespace root_in_interval_l3036_303622

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem root_in_interval :
  ∃! r : ℝ, r ∈ Set.Ioo (1/4 : ℝ) (1/2 : ℝ) ∧ f r = 0 :=
by
  sorry

end root_in_interval_l3036_303622


namespace seven_eighths_of_64_l3036_303663

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by sorry

end seven_eighths_of_64_l3036_303663


namespace triangle_reconstruction_l3036_303648

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the incenter of a triangle -/
def incenter (t : Triangle) : Point := sorry

/-- Represents the foot of altitude from C to AB -/
def altitudeFootC (t : Triangle) : Point := sorry

/-- Represents the C-excenter of a triangle -/
def excenterC (t : Triangle) : Point := sorry

/-- Theorem: Given the incenter, foot of altitude from C, and C-excenter, 
    a unique triangle can be reconstructed -/
theorem triangle_reconstruction 
  (I : Point) (H : Point) (I_c : Point) : 
  ∃! (t : Triangle), 
    incenter t = I ∧ 
    altitudeFootC t = H ∧ 
    excenterC t = I_c := by sorry

end triangle_reconstruction_l3036_303648


namespace book_distribution_l3036_303656

theorem book_distribution (m : ℕ) (x y : ℕ) : 
  (y - 14 = m * x) →  -- If each child gets m books, 14 books are left
  (y + 3 = 9 * x) →   -- If each child gets 9 books, the last child gets 6 books
  (x = 17 ∧ y = 150) := by
  sorry

end book_distribution_l3036_303656


namespace i_to_2016_l3036_303614

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem i_to_2016 : i^2016 = 1 := by
  sorry

end i_to_2016_l3036_303614


namespace value_of_b_l3036_303670

theorem value_of_b (x y : ℝ) : 
  x = (1 - Real.sqrt 3) / (1 + Real.sqrt 3) →
  y = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) →
  2 * x^2 - 3 * x * y + 2 * y^2 = 25 :=
by sorry

end value_of_b_l3036_303670


namespace line_through_point_equal_intercepts_l3036_303630

/-- A line passing through (1, 2) with equal X and Y intercepts has equation 2x - y = 0 or x + y - 3 = 0 -/
theorem line_through_point_equal_intercepts :
  ∀ (a b c : ℝ),
    (a ≠ 0 ∧ b ≠ 0) →
    (a * 1 + b * 2 + c = 0) →  -- Line passes through (1, 2)
    ((-c/a) = (-c/b)) →        -- Equal X and Y intercepts
    ((a = 2 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -3)) := by
  sorry


end line_through_point_equal_intercepts_l3036_303630


namespace sequence_term_l3036_303609

theorem sequence_term (a : ℕ → ℝ) (h : ∀ n, a n = Real.sqrt (3 * n - 1)) :
  a 7 = 2 * Real.sqrt 5 := by
  sorry

end sequence_term_l3036_303609


namespace equation_solution_l3036_303697

theorem equation_solution (x : ℝ) : 
  (1 - |Real.cos x|) / (1 + |Real.cos x|) = Real.sin x → 
  (∃ k : ℤ, x = k * Real.pi ∨ x = 2 * k * Real.pi + Real.pi / 2) := by
  sorry

end equation_solution_l3036_303697


namespace thousand_factorial_zeroes_l3036_303649

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- n! is the product of integers from 1 to n -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem thousand_factorial_zeroes :
  trailingZeroes 1000 = 249 :=
sorry

end thousand_factorial_zeroes_l3036_303649


namespace circumscribed_circle_radius_of_specific_trapezoid_l3036_303674

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  a : ℝ  -- Length of the longer base
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumscribedCircleRadius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed circle of the given isosceles trapezoid is 85/8 -/
theorem circumscribed_circle_radius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := { a := 21, b := 9, h := 8 }
  circumscribedCircleRadius t = 85 / 8 := by
  sorry

end circumscribed_circle_radius_of_specific_trapezoid_l3036_303674


namespace balloon_permutations_l3036_303637

def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

theorem balloon_permutations :
  balloon_arrangements = 1260 := by
  sorry

end balloon_permutations_l3036_303637


namespace probability_three_students_l3036_303696

/-- The probability of having students participate on both Saturday and Sunday -/
def probability_both_days (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (2^n - 2) / 2^n

theorem probability_three_students :
  probability_both_days 3 = 3/4 := by
  sorry

end probability_three_students_l3036_303696


namespace max_n_for_factorable_quadratic_l3036_303682

/-- 
Given a quadratic expression 6x^2 + nx + 108 that can be factored as the product 
of two linear factors with integer coefficients, the maximum possible value of n is 649.
-/
theorem max_n_for_factorable_quadratic : 
  ∀ n : ℤ, 
  (∃ A B : ℤ, 6 * A * B = 108 ∧ 6 * B + A = n) → 
  n ≤ 649 :=
by sorry

end max_n_for_factorable_quadratic_l3036_303682


namespace year_spans_weeks_l3036_303692

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of days in a common year -/
def daysInCommonYear : ℕ := 365

/-- Represents the number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- Represents the minimum number of days a week must have in a year to be counted -/
def minDaysInWeekForYear : ℕ := 6

/-- Definition of how many weeks a year can span -/
def weeksInYear : Set ℕ := {53, 54}

/-- Theorem stating the number of weeks a year can span -/
theorem year_spans_weeks : 
  ∀ (year : ℕ), 
    (year = daysInCommonYear ∨ year = daysInLeapYear) → 
    ∃ (weeks : ℕ), weeks ∈ weeksInYear ∧ 
      (weeks - 1) * daysInWeek + minDaysInWeekForYear ≤ year ∧
      year < (weeks + 1) * daysInWeek :=
sorry

end year_spans_weeks_l3036_303692


namespace function_identity_l3036_303612

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem function_identity (f : ℕ+ → ℕ+) : 
  (∀ m n : ℕ+, is_divisible (m^2 + f n) (m * f m + n)) → 
  (∀ n : ℕ+, f n = n) :=
by sorry

end function_identity_l3036_303612


namespace expression_evaluation_l3036_303688

theorem expression_evaluation (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x^2 / y^3) / (y / x) = 3 / 16 := by
  sorry

end expression_evaluation_l3036_303688


namespace work_completion_time_l3036_303618

/-- The time it takes for worker A to complete the work alone -/
def time_A : ℝ := 12

/-- The time it takes for workers A and B to complete the work together -/
def time_AB : ℝ := 7.2

/-- The time it takes for worker B to complete the work alone -/
def time_B : ℝ := 18

/-- Theorem stating that given the time for A and the time for A and B together,
    we can prove that B takes 18 days to complete the work alone -/
theorem work_completion_time :
  (1 / time_A + 1 / time_B = 1 / time_AB) → time_B = 18 := by
  sorry

end work_completion_time_l3036_303618


namespace expression_evaluation_l3036_303639

theorem expression_evaluation :
  60 + 120 / 15 + 25 * 16 - 220 - 420 / 7 + 3^2 = 197 := by
  sorry

end expression_evaluation_l3036_303639


namespace arithmetic_sequence_sum_l3036_303652

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 6 + a 7 + a 8 + a 9 = 450) :
  a 3 + a 11 = 180 := by
sorry

end arithmetic_sequence_sum_l3036_303652


namespace parabola_focus_point_slope_l3036_303615

/-- The slope of line AF for a parabola y² = 4x with focus F(1,0) and point A on the parabola -/
theorem parabola_focus_point_slope (A : ℝ × ℝ) : 
  A.1 > 0 → -- A is in the first quadrant
  A.2 > 0 →
  A.1 + 1 = 5 → -- distance from A to directrix x = -1 is 5
  A.2^2 = 4 * A.1 → -- A is on the parabola y² = 4x
  (A.2 - 0) / (A.1 - 1) = Real.sqrt 3 := by sorry

end parabola_focus_point_slope_l3036_303615


namespace problem_solution_l3036_303678

-- Define the ⊗ operation
def otimes (a b : ℕ) : ℕ := sorry

-- Define the main property of ⊗
axiom otimes_prop (a b c : ℕ) : otimes a b = c ↔ a^c = b

theorem problem_solution :
  (∀ x, otimes 3 81 = x → x = 4) ∧
  (∀ a b c, otimes 3 5 = a → otimes 3 6 = b → otimes 3 10 = c → a < b ∧ b < c) :=
by sorry

end problem_solution_l3036_303678


namespace polynomial_equation_solution_l3036_303600

theorem polynomial_equation_solution : 
  ∃ x : ℝ, ((x^3 * 0.76^3 - 0.008) / (x^2 * 0.76^2 + x * 0.76 * 0.2 + 0.04) = 0) ∧ 
  (abs (x - 0.262) < 0.001) := by
  sorry

end polynomial_equation_solution_l3036_303600


namespace solution_set_f_range_of_a_solution_set_eq_interval_l3036_303683

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 1|

-- Theorem for part (I)
theorem solution_set_f (x : ℝ) : 
  f x ≥ 4*x + 3 ↔ x ∈ Set.Iic (-3/7) := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * f x ≥ 3*a^2 - a - 1) → 
  a ∈ Set.Icc (-1) (4/3) := by sorry

-- Define the set of solutions for part (I)
def solution_set : Set ℝ := {x : ℝ | f x ≥ 4*x + 3}

-- Theorem stating that the solution set is equal to (-∞, -3/7]
theorem solution_set_eq_interval : 
  solution_set = Set.Iic (-3/7) := by sorry

end solution_set_f_range_of_a_solution_set_eq_interval_l3036_303683
