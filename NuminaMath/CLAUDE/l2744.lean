import Mathlib

namespace garage_bikes_l2744_274404

/-- Given a number of wheels and the number of wheels required per bike, 
    calculate the number of bikes that can be assembled -/
def bikes_assembled (total_wheels : ℕ) (wheels_per_bike : ℕ) : ℕ :=
  total_wheels / wheels_per_bike

theorem garage_bikes : bikes_assembled 14 2 = 7 := by
  sorry

end garage_bikes_l2744_274404


namespace arrangements_with_constraints_l2744_274466

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

def doubly_adjacent_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 2)

theorem arrangements_with_constraints (n : ℕ) (h : n = 5) : 
  total_arrangements n - 2 * adjacent_arrangements n + doubly_adjacent_arrangements n = 36 := by
  sorry

#check arrangements_with_constraints

end arrangements_with_constraints_l2744_274466


namespace intersection_and_complement_l2744_274458

def A : Set ℝ := {x : ℝ | -4 < x^2 - 5*x + 2 ∧ x^2 - 5*x + 2 < 26}
def B : Set ℝ := {x : ℝ | -x^2 + 4*x - 3 < 0}

theorem intersection_and_complement :
  (A ∩ B = {x : ℝ | (-3 < x ∧ x < 1) ∨ (3 < x ∧ x < 8)}) ∧
  (Set.compl (A ∩ B) = {x : ℝ | x ≤ -3 ∨ (1 ≤ x ∧ x ≤ 3) ∨ x ≥ 8}) := by
  sorry

end intersection_and_complement_l2744_274458


namespace troll_ratio_l2744_274491

/-- The number of trolls hiding by the path in the forest -/
def trolls_by_path : ℕ := 6

/-- The total number of trolls counted -/
def total_trolls : ℕ := 33

/-- The number of trolls hiding under the bridge -/
def trolls_under_bridge : ℕ := 18

/-- The number of trolls hiding in the plains -/
def trolls_in_plains : ℕ := trolls_under_bridge / 2

theorem troll_ratio : 
  trolls_by_path + trolls_under_bridge + trolls_in_plains = total_trolls ∧ 
  trolls_under_bridge / trolls_by_path = 3 := by
  sorry

end troll_ratio_l2744_274491


namespace no_solution_to_inequality_l2744_274479

theorem no_solution_to_inequality : 
  ¬ ∃ x : ℝ, -2 < (x^2 - 10*x + 9) / (x^2 - 4*x + 8) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 8) < 2 :=
by sorry

end no_solution_to_inequality_l2744_274479


namespace four_digit_integer_problem_l2744_274481

theorem four_digit_integer_problem (a b c d : ℕ) : 
  a ≠ 0 ∧ 
  a + b + c + d = 16 ∧ 
  b + c = 11 ∧ 
  a - d = 3 ∧ 
  (1000 * a + 100 * b + 10 * c + d) % 11 = 0 →
  1000 * a + 100 * b + 10 * c + d = 4714 := by
sorry

end four_digit_integer_problem_l2744_274481


namespace complex_number_identity_l2744_274433

theorem complex_number_identity (a b c : ℂ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : a + b + c = 15) 
  (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) : 
  (a^3 + b^3 + c^3) / (a*b*c) = 18 := by
sorry

end complex_number_identity_l2744_274433


namespace smallest_distance_between_complex_numbers_l2744_274407

theorem smallest_distance_between_complex_numbers (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 2*Complex.I) = 2)
  (hw : Complex.abs (w - 5 - 6*Complex.I) = 2) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 113 - 4 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' + 2 + 2*Complex.I) = 2 →
      Complex.abs (w' - 5 - 6*Complex.I) = 2 →
      Complex.abs (z' - w') ≥ min_dist := by
  sorry

end smallest_distance_between_complex_numbers_l2744_274407


namespace salary_reduction_percentage_l2744_274415

theorem salary_reduction_percentage (S : ℝ) (P : ℝ) (h : S > 0) :
  2 * (S - (P / 100 * S)) = S → P = 50 := by
  sorry

end salary_reduction_percentage_l2744_274415


namespace ab_value_l2744_274470

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a^4 + b^4 = 31/16) : 
  a * b = Real.sqrt (33/32) := by
sorry

end ab_value_l2744_274470


namespace two_people_two_rooms_probability_two_people_two_rooms_probability_proof_l2744_274445

/-- The probability that two people randomly checking into two rooms will each occupy one room -/
theorem two_people_two_rooms_probability : ℝ :=
  1/2

/-- Proof that the probability of two people randomly checking into two rooms 
    and each occupying one room is 1/2 -/
theorem two_people_two_rooms_probability_proof : 
  two_people_two_rooms_probability = 1/2 := by
  sorry

end two_people_two_rooms_probability_two_people_two_rooms_probability_proof_l2744_274445


namespace other_root_of_quadratic_l2744_274426

theorem other_root_of_quadratic (x : ℚ) :
  (7 * x^2 - 3 * x = 10) ∧ (7 * (-2)^2 - 3 * (-2) = 10) →
  (7 * (5/7)^2 - 3 * (5/7) = 10) :=
by sorry

end other_root_of_quadratic_l2744_274426


namespace smallest_valid_club_size_l2744_274423

def is_valid_club_size (N : ℕ) : Prop :=
  N < 50 ∧
  ((N - 5) % 6 = 0 ∨ (N - 5) % 7 = 0) ∧
  N % 8 = 7

theorem smallest_valid_club_size :
  ∀ n : ℕ, is_valid_club_size n → n ≥ 47 :=
by sorry

end smallest_valid_club_size_l2744_274423


namespace add_2687_minutes_to_7am_l2744_274475

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem add_2687_minutes_to_7am (start : Time) (h : start.hours = 7 ∧ start.minutes = 0) :
  addMinutes start 2687 = { hours := 3, minutes := 47, h_valid := sorry, m_valid := sorry } :=
sorry

end add_2687_minutes_to_7am_l2744_274475


namespace typist_salary_problem_l2744_274443

/-- Proves that if a salary is first increased by 10% and then decreased by 5%, 
    resulting in Rs. 4180, then the original salary must be Rs. 4000. -/
theorem typist_salary_problem (x : ℝ) : 
  (x * 1.1 * 0.95 = 4180) → x = 4000 := by
  sorry

end typist_salary_problem_l2744_274443


namespace triangle_4_6_9_l2744_274471

/-- Defines whether three given lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that lengths 4, 6, and 9 can form a triangle -/
theorem triangle_4_6_9 :
  can_form_triangle 4 6 9 := by
  sorry

end triangle_4_6_9_l2744_274471


namespace metal_weight_in_compound_l2744_274422

/-- The molecular weight of the metal element in a compound with formula (OH)2 -/
def metal_weight (total_weight : ℝ) : ℝ :=
  total_weight - 2 * (16 + 1)

/-- Theorem: The molecular weight of the metal element in a compound with formula (OH)2
    and total molecular weight of 171 g/mol is 171 - 2 * (16 + 1) g/mol -/
theorem metal_weight_in_compound : metal_weight 171 = 137 := by
  sorry

end metal_weight_in_compound_l2744_274422


namespace sum_of_twelve_terms_special_case_l2744_274485

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- The common difference
  h : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_of_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

theorem sum_of_twelve_terms_special_case (seq : ArithmeticSequence) 
  (h₁ : seq.a 5 = 1)
  (h₂ : seq.a 17 = 18) :
  sum_of_terms seq 12 = 37.5 := by
  sorry

end sum_of_twelve_terms_special_case_l2744_274485


namespace original_house_price_l2744_274461

/-- Given a house that increases in value by 25% and is then sold to cover 25% of a $500,000 new house,
    prove that the original purchase price of the first house was $100,000. -/
theorem original_house_price (original_price : ℝ) : 
  (original_price * 1.25 = 500000 * 0.25) → original_price = 100000 := by
  sorry

end original_house_price_l2744_274461


namespace eggs_needed_is_84_l2744_274498

/-- Represents the number of eggs in an omelette type -/
inductive OmeletteType
| ThreeEgg
| FourEgg

/-- Represents an hour's worth of orders -/
structure HourlyOrder where
  customerCount : Nat
  omeletteType : OmeletteType

/-- Calculates the total number of eggs needed for all omelettes -/
def totalEggsNeeded (orders : List HourlyOrder) : Nat :=
  orders.foldl (fun acc order =>
    acc + order.customerCount * match order.omeletteType with
      | OmeletteType.ThreeEgg => 3
      | OmeletteType.FourEgg => 4
  ) 0

theorem eggs_needed_is_84 (orders : List HourlyOrder) 
  (h1 : orders = [
    ⟨5, OmeletteType.ThreeEgg⟩, 
    ⟨7, OmeletteType.FourEgg⟩,
    ⟨3, OmeletteType.ThreeEgg⟩,
    ⟨8, OmeletteType.FourEgg⟩
  ]) : 
  totalEggsNeeded orders = 84 := by
  sorry

#eval totalEggsNeeded [
  ⟨5, OmeletteType.ThreeEgg⟩, 
  ⟨7, OmeletteType.FourEgg⟩,
  ⟨3, OmeletteType.ThreeEgg⟩,
  ⟨8, OmeletteType.FourEgg⟩
]

end eggs_needed_is_84_l2744_274498


namespace quadratic_inequality_solution_l2744_274494

theorem quadratic_inequality_solution (m n : ℝ) :
  (∀ x : ℝ, x^2 + m*x + n < 0 ↔ -1 < x ∧ x < 3) →
  m + n = -1 :=
by sorry

end quadratic_inequality_solution_l2744_274494


namespace remove_six_maximizes_probability_l2744_274487

def original_list : List ℤ := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def is_valid_pair (x y : ℤ) : Prop :=
  x ∈ original_list ∧ y ∈ original_list ∧ x ≠ y ∧ x + y = 12

def count_valid_pairs (removed : ℤ) : ℕ :=
  (original_list.filter (λ x => x ≠ removed)).length.choose 2

theorem remove_six_maximizes_probability :
  ∀ n ∈ original_list, count_valid_pairs 6 ≥ count_valid_pairs n :=
by sorry

end remove_six_maximizes_probability_l2744_274487


namespace isosceles_triangle_area_l2744_274410

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  altitude : ℝ
  perimeter : ℝ

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 :=
by sorry

end isosceles_triangle_area_l2744_274410


namespace lunch_scores_pigeonhole_l2744_274459

theorem lunch_scores_pigeonhole (n : ℕ) (scores : Fin n → ℕ) 
  (h1 : ∀ i : Fin n, scores i < n) : 
  ∃ i j : Fin n, i ≠ j ∧ scores i = scores j :=
by
  sorry

end lunch_scores_pigeonhole_l2744_274459


namespace hyperbola_real_axis_length_l2744_274413

/-- A hyperbola with center at the origin and foci on the x-axis -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool

/-- The length of a line segment -/
def length (a b : ℝ × ℝ) : ℝ := sorry

/-- The real axis of a hyperbola -/
def real_axis (h : Hyperbola) : ℝ := sorry

theorem hyperbola_real_axis_length
  (C : Hyperbola)
  (h_center : C.center = (0, 0))
  (h_foci : C.foci_on_x_axis = true)
  (A B : ℝ × ℝ)
  (h_intersect : A.1 = -4 ∧ B.1 = -4)
  (h_distance : length A B = 4) :
  real_axis C = 4 := by sorry

end hyperbola_real_axis_length_l2744_274413


namespace smallest_x_absolute_value_equation_l2744_274414

theorem smallest_x_absolute_value_equation :
  ∃ x : ℝ, (∀ y : ℝ, |y + 3| = 15 → x ≤ y) ∧ |x + 3| = 15 :=
by sorry

end smallest_x_absolute_value_equation_l2744_274414


namespace circle_tangent_radius_l2744_274449

/-- Given a system of equations describing the geometry of two circles with a common tangent,
    prove that the radius r of one circle is equal to 2. -/
theorem circle_tangent_radius (a r : ℝ) : 
  ((4 - r)^2 + a^2 = (4 + r)^2) ∧ 
  (r^2 + a^2 = (8 - r)^2) → 
  r = 2 := by
  sorry

end circle_tangent_radius_l2744_274449


namespace f_strictly_increasing_l2744_274453

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- State the theorem
theorem f_strictly_increasing :
  (∀ x y, x < y ∧ ((x < -1/3 ∧ y ≤ -1/3) ∨ (x ≥ 1 ∧ y > 1)) → f x < f y) :=
by sorry

end f_strictly_increasing_l2744_274453


namespace specific_tank_insulation_cost_l2744_274425

/-- The cost to insulate a rectangular tank -/
def insulation_cost (length width height cost_per_sqft : ℝ) : ℝ :=
  (2 * (length * width + length * height + width * height)) * cost_per_sqft

/-- Theorem: The cost to insulate a specific rectangular tank -/
theorem specific_tank_insulation_cost :
  insulation_cost 4 5 3 20 = 1880 := by
  sorry

end specific_tank_insulation_cost_l2744_274425


namespace union_A_B_complement_A_l2744_274409

open Set

-- Define the universe set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 < 4}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

-- Theorem for the complement of A with respect to U
theorem complement_A : (U \ A) = {x : ℝ | x < -1 ∨ x > 3} := by sorry

end union_A_B_complement_A_l2744_274409


namespace water_mass_in_range_l2744_274427

/-- Represents the thermodynamic properties of a substance -/
structure ThermodynamicProperties where
  specific_heat_capacity : Real
  specific_latent_heat : Real

/-- Represents the initial state of a substance -/
structure InitialState where
  mass : Real
  temperature : Real

/-- Calculates the range of added water mass given the initial conditions and final temperature -/
def calculate_water_mass_range (ice_props : ThermodynamicProperties)
                               (water_props : ThermodynamicProperties)
                               (ice_initial : InitialState)
                               (water_initial : InitialState)
                               (final_temp : Real) : Set Real :=
  sorry

/-- Theorem stating that the mass of added water lies within the calculated range -/
theorem water_mass_in_range :
  let ice_props : ThermodynamicProperties := {
    specific_heat_capacity := 2100,
    specific_latent_heat := 3.3e5
  }
  let water_props : ThermodynamicProperties := {
    specific_heat_capacity := 4200,
    specific_latent_heat := 0
  }
  let ice_initial : InitialState := {
    mass := 0.1,
    temperature := -5
  }
  let water_initial : InitialState := {
    mass := 0,  -- mass to be determined
    temperature := 10
  }
  let final_temp : Real := 0
  let water_mass_range := calculate_water_mass_range ice_props water_props ice_initial water_initial final_temp
  ∀ m ∈ water_mass_range, 0.0028 ≤ m ∧ m ≤ 0.8119 :=
by sorry

end water_mass_in_range_l2744_274427


namespace x_squared_minus_y_squared_l2744_274488

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9/17) 
  (h2 : x - y = 1/51) : 
  x^2 - y^2 = 9/867 := by
sorry

end x_squared_minus_y_squared_l2744_274488


namespace no_quadratic_factor_l2744_274421

def p (x : ℝ) : ℝ := x^4 - 6*x^2 + 25

def q₁ (x : ℝ) : ℝ := x^2 - 3*x + 4
def q₂ (x : ℝ) : ℝ := x^2 - 4
def q₃ (x : ℝ) : ℝ := x^2 + 3
def q₄ (x : ℝ) : ℝ := x^2 + 3*x - 4

theorem no_quadratic_factor :
  (∀ x, p x ≠ 0 → q₁ x ≠ 0) ∧
  (∀ x, p x ≠ 0 → q₂ x ≠ 0) ∧
  (∀ x, p x ≠ 0 → q₃ x ≠ 0) ∧
  (∀ x, p x ≠ 0 → q₄ x ≠ 0) :=
by sorry

end no_quadratic_factor_l2744_274421


namespace number_of_baskets_l2744_274440

theorem number_of_baskets (green_per_basket : ℕ) (total_green : ℕ) (h1 : green_per_basket = 2) (h2 : total_green = 14) :
  total_green / green_per_basket = 7 := by
sorry

end number_of_baskets_l2744_274440


namespace water_tank_capacity_l2744_274496

theorem water_tank_capacity (initial_fraction : ℚ) (added_gallons : ℕ) (total_capacity : ℕ) : 
  initial_fraction = 1/3 →
  added_gallons = 16 →
  initial_fraction * total_capacity + added_gallons = total_capacity →
  total_capacity = 24 :=
by
  sorry

end water_tank_capacity_l2744_274496


namespace polynomial_factorization_l2744_274436

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x - 35 = (x - 7)*(x + 5)) → m = -2 := by
  sorry

end polynomial_factorization_l2744_274436


namespace matrix_multiplication_example_l2744_274472

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 3; 7, -1]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, -5; 0, 4]
  A * B = !![2, 2; 7, -39] := by
sorry

end matrix_multiplication_example_l2744_274472


namespace odd_function_and_inequality_l2744_274463

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 4 / (2 * a^x + a)

theorem odd_function_and_inequality 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x, f a x = -f a (-x)) -- odd function condition
  (h4 : ∀ x, f a x ∈ Set.univ) -- defined on (-∞, +∞)
  : 
  (a = 2) ∧ 
  (∀ t : ℝ, (∀ x ∈ Set.Ioc 0 1, t * f a x ≥ 2^x - 2) ↔ t ≥ 0) :=
sorry

end odd_function_and_inequality_l2744_274463


namespace sum_32_45_base5_l2744_274483

/-- Converts a base 10 number to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number --/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_32_45_base5 :
  toBase5 (32 + 45) = [3, 0, 2] :=
sorry

end sum_32_45_base5_l2744_274483


namespace total_pepper_weight_l2744_274405

-- Define the weights of green and red peppers
def green_peppers : ℝ := 0.33
def red_peppers : ℝ := 0.33

-- Theorem stating the total weight of peppers
theorem total_pepper_weight : green_peppers + red_peppers = 0.66 := by
  sorry

end total_pepper_weight_l2744_274405


namespace ratio_a_over_4_to_b_over_3_l2744_274478

theorem ratio_a_over_4_to_b_over_3 (a b c : ℝ) 
  (h1 : 3 * a^2 = 4 * b^2)
  (h2 : a * b * (c^2 + 2*c + 1) ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a = 2*c^2 + 3*c + c^(1/2))
  (h5 : b = c^2 + 5*c - c^(3/2)) :
  (a / 4) / (b / 3) = Real.sqrt 3 / 2 := by
sorry

end ratio_a_over_4_to_b_over_3_l2744_274478


namespace prime_conditions_theorem_l2744_274490

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_conditions (A : ℕ) : Prop :=
  is_prime A ∧
  A < 100 ∧
  is_prime (A + 10) ∧
  is_prime (A - 20) ∧
  is_prime (A + 30) ∧
  is_prime (A + 60) ∧
  is_prime (A + 70)

theorem prime_conditions_theorem :
  ∀ A : ℕ, satisfies_conditions A ↔ (A = 37 ∨ A = 43 ∨ A = 79) :=
by sorry

end prime_conditions_theorem_l2744_274490


namespace mean_of_readings_l2744_274493

def readings : List ℝ := [2, 2.1, 2, 2.2]

theorem mean_of_readings (x : ℝ) (mean : ℝ) : 
  readings.length = 4 →
  mean = (readings.sum + x) / 5 := by
  sorry

end mean_of_readings_l2744_274493


namespace infinite_decimal_is_rational_l2744_274448

/-- Given an infinite decimal T = 0.a₁a₂a₃..., where aₙ is the remainder when n² is divided by 10,
    prove that T is equal to 166285490 / 1111111111. -/
theorem infinite_decimal_is_rational :
  let a : ℕ → ℕ := λ n => n^2 % 10
  let T : ℝ := ∑' n, (a n : ℝ) / 10^(n + 1)
  T = 166285490 / 1111111111 :=
sorry

end infinite_decimal_is_rational_l2744_274448


namespace highest_consecutive_number_l2744_274474

theorem highest_consecutive_number (n : ℤ) (h1 : n - 3 + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) = 33 * 7) :
  n + 3 = 36 := by
  sorry

end highest_consecutive_number_l2744_274474


namespace means_inequality_l2744_274480

theorem means_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ (a * b * c) ^ (1/3) > 3 * a * b * c / (a * b + b * c + c * a) :=
sorry

end means_inequality_l2744_274480


namespace book_thickness_theorem_l2744_274460

/-- Calculates the number of pages per inch of thickness for a stack of books --/
def pages_per_inch (num_books : ℕ) (avg_pages : ℕ) (total_thickness : ℕ) : ℕ :=
  (num_books * avg_pages) / total_thickness

/-- Theorem: Given a stack of 6 books with an average of 160 pages each and a total thickness of 12 inches,
    the number of pages per inch of thickness is 80. --/
theorem book_thickness_theorem :
  pages_per_inch 6 160 12 = 80 := by
  sorry

end book_thickness_theorem_l2744_274460


namespace triangle_side_length_l2744_274432

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  a = 2 →
  B = π / 3 →
  c = 3 →
  b = Real.sqrt 7 :=
by
  sorry

end triangle_side_length_l2744_274432


namespace lawn_care_supplies_cost_l2744_274484

/-- The total cost of supplies for a lawn care company -/
theorem lawn_care_supplies_cost 
  (num_blades : ℕ) 
  (blade_cost : ℕ) 
  (string_cost : ℕ) : 
  num_blades = 4 → 
  blade_cost = 8 → 
  string_cost = 7 → 
  num_blades * blade_cost + string_cost = 39 :=
by
  sorry

end lawn_care_supplies_cost_l2744_274484


namespace no_overlap_for_y_l2744_274424

theorem no_overlap_for_y (y : ℝ) : 
  200 ≤ y ∧ y ≤ 300 → 
  ⌊Real.sqrt y⌋ = 16 → 
  ⌊Real.sqrt (50 * y)⌋ ≠ 226 := by
sorry

end no_overlap_for_y_l2744_274424


namespace smallest_root_of_equation_l2744_274486

theorem smallest_root_of_equation (x : ℝ) : 
  (|x - 1| / x^2 = 6) → (x = -1/2 ∨ x = 1/3) ∧ (-1/2 < 1/3) := by
  sorry

end smallest_root_of_equation_l2744_274486


namespace total_amount_is_15_l2744_274431

-- Define the shares of w, x, and y
def w_share : ℝ := 10
def x_share : ℝ := w_share * 0.3
def y_share : ℝ := w_share * 0.2

-- Define the total amount
def total_amount : ℝ := w_share + x_share + y_share

-- Theorem statement
theorem total_amount_is_15 : total_amount = 15 := by
  sorry

end total_amount_is_15_l2744_274431


namespace inverse_composition_problem_l2744_274418

def f : Fin 6 → Fin 6
| 1 => 4
| 2 => 5
| 3 => 3
| 4 => 2
| 5 => 1
| 6 => 6

theorem inverse_composition_problem (h : Function.Bijective f) :
  (Function.invFun f) ((Function.invFun f) ((Function.invFun f) 2)) = 5 := by
  sorry

end inverse_composition_problem_l2744_274418


namespace max_sum_of_roots_l2744_274454

theorem max_sum_of_roots (c b : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - c*x + b = 0 ∧ y^2 - c*y + b = 0 ∧ x - y = 1) →
  c ≤ 11 :=
by sorry

end max_sum_of_roots_l2744_274454


namespace existence_of_bounded_difference_l2744_274469

theorem existence_of_bounded_difference (n : ℕ) (x : Fin n → ℝ) 
  (h_n : n ≥ 3) 
  (h_pos : ∀ i, x i > 0) 
  (h_distinct : ∀ i j, i ≠ j → x i ≠ x j) : 
  ∃ i j, i ≠ j ∧ 
    0 < (x i - x j) / (1 + x i * x j) ∧ 
    (x i - x j) / (1 + x i * x j) < Real.tan (π / (2 * (n - 1))) := by
  sorry

end existence_of_bounded_difference_l2744_274469


namespace right_triangle_area_l2744_274477

theorem right_triangle_area (a b : ℝ) (h_a : a = 25) (h_b : b = 20) :
  (1 / 2) * a * b = 250 :=
by sorry

end right_triangle_area_l2744_274477


namespace square_of_cube_zero_matrix_l2744_274450

theorem square_of_cube_zero_matrix (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 3 = 0) : A ^ 2 = 0 := by
  sorry

end square_of_cube_zero_matrix_l2744_274450


namespace square_area_ratio_l2744_274417

theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (3 * y)^2 / (9 * y)^2 = 1 / 9 := by sorry

end square_area_ratio_l2744_274417


namespace school_population_l2744_274455

theorem school_population (total boys girls : ℕ) : 
  (total = boys + girls) →
  (boys = 50 → girls = total / 2) →
  total = 100 := by
sorry

end school_population_l2744_274455


namespace inequality_proof_l2744_274497

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end inequality_proof_l2744_274497


namespace trig_identity_l2744_274482

theorem trig_identity : 
  Real.sin (63 * π / 180) * Real.cos (18 * π / 180) + 
  Real.cos (63 * π / 180) * Real.cos (108 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end trig_identity_l2744_274482


namespace inscribed_square_area_l2744_274441

/-- The area of a square inscribed in the ellipse x^2/5 + y^2/10 = 1, with its diagonals parallel to the coordinate axes, is 40/3. -/
theorem inscribed_square_area (x y : ℝ) :
  (x^2 / 5 + y^2 / 10 = 1) →  -- ellipse equation
  (∃ (a : ℝ), x = a ∧ y = a) →  -- square vertices on the ellipse
  (40 : ℝ) / 3 = 4 * x^2 := by
  sorry

end inscribed_square_area_l2744_274441


namespace silver_status_families_l2744_274451

def fundraiser (bronze silver gold : ℕ) : ℕ := 
  25 * bronze + 50 * silver + 100 * gold

theorem silver_status_families : 
  ∃ (silver : ℕ), 
    fundraiser 10 silver 1 = 700 ∧ 
    silver = 7 :=
by
  sorry

end silver_status_families_l2744_274451


namespace mango_rate_calculation_l2744_274499

theorem mango_rate_calculation (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (total_paid : ℕ) :
  apple_quantity = 8 →
  apple_rate = 70 →
  mango_quantity = 9 →
  total_paid = 1055 →
  (total_paid - apple_quantity * apple_rate) / mango_quantity = 55 :=
by
  sorry

end mango_rate_calculation_l2744_274499


namespace hyperbola_and_related_ellipse_l2744_274446

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, prove its asymptotes and related ellipse equations -/
theorem hyperbola_and_related_ellipse 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_imag_axis : b = 1) 
  (h_focal_dist : 2 * Real.sqrt 3 = 2 * Real.sqrt (a^2 + b^2)) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = Real.sqrt 2 / 2 * x) ∧ 
                   (∀ x, f (-x) = -f x) ∧ 
                   (∀ x y, y = f x ∨ y = -f x → x^2/a^2 - y^2/b^2 = 1)) ∧
  (∀ x y, x^2/3 + y^2 = 1 → 
    ∃ (t : ℝ), x = Real.sqrt 3 * Real.cos t ∧ y = Real.sin t) := by
  sorry

end hyperbola_and_related_ellipse_l2744_274446


namespace three_sequences_comparison_l2744_274420

theorem three_sequences_comparison 
  (a b c : ℕ → ℕ) : 
  ∃ m n : ℕ, m ≠ n ∧ 
    a m ≥ a n ∧ 
    b m ≥ b n ∧ 
    c m ≥ c n :=
by sorry

end three_sequences_comparison_l2744_274420


namespace hyperbola_standard_form_l2744_274444

/-- The standard form of a hyperbola with foci on the x-axis -/
def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The relationship between a, b, and c in a hyperbola -/
def hyperbola_relation (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem hyperbola_standard_form :
  ∀ (a b : ℝ),
    a > 0 →
    b > 0 →
    hyperbola_relation a b (Real.sqrt 6) →
    hyperbola_equation a b (-5) 2 →
    hyperbola_equation (Real.sqrt 5) 1 = hyperbola_equation a b :=
by sorry

end hyperbola_standard_form_l2744_274444


namespace rons_height_l2744_274476

theorem rons_height (dean_height ron_height water_depth : ℝ) : 
  water_depth = 2 * dean_height →
  dean_height = ron_height - 8 →
  water_depth = 12 →
  ron_height = 14 := by
  sorry

end rons_height_l2744_274476


namespace purely_imaginary_fraction_l2744_274401

theorem purely_imaginary_fraction (a : ℝ) : 
  (∃ k : ℝ, (a - I) / (1 + I) = k * I) → a = -1 :=
by
  sorry

end purely_imaginary_fraction_l2744_274401


namespace number_equation_solution_l2744_274452

theorem number_equation_solution : ∃ x : ℝ, (4 * x - 7 = 13) ∧ (x = 5) := by
  sorry

end number_equation_solution_l2744_274452


namespace arithmetic_sequence_common_difference_l2744_274408

/-- 
Given an arithmetic sequence with first term 5, last term 50, and sum of all terms 330,
prove that the common difference is 45/11.
-/
theorem arithmetic_sequence_common_difference 
  (a₁ : ℚ) (aₙ : ℚ) (S : ℚ) (n : ℕ) (d : ℚ) :
  a₁ = 5 →
  aₙ = 50 →
  S = 330 →
  S = n / 2 * (a₁ + aₙ) →
  aₙ = a₁ + (n - 1) * d →
  d = 45 / 11 := by
  sorry

end arithmetic_sequence_common_difference_l2744_274408


namespace permutation_residue_system_bound_l2744_274434

/-- A permutation of (1, 2, ..., n) -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The set {pᵢ + i | 1 ≤ i ≤ n} is a complete residue system modulo n -/
def IsSumCompleteResidue (n : ℕ) (p : Permutation n) : Prop :=
  ∀ k : Fin n, ∃ i : Fin n, (p i + i : ℕ) % n = k

/-- The set {pᵢ - i | 1 ≤ i ≤ n} is a complete residue system modulo n -/
def IsDiffCompleteResidue (n : ℕ) (p : Permutation n) : Prop :=
  ∀ k : Fin n, ∃ i : Fin n, ((p i : ℕ) - (i : ℕ) + n) % n = k

/-- Main theorem: If n satisfies the conditions, then n ≥ 4 -/
theorem permutation_residue_system_bound (n : ℕ) :
  (∃ p : Permutation n, IsSumCompleteResidue n p ∧ IsDiffCompleteResidue n p) →
  n ≥ 4 := by
  sorry

end permutation_residue_system_bound_l2744_274434


namespace angle_inequality_l2744_274438

theorem angle_inequality (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ ≥ 0) →
  Real.pi / 12 ≤ θ ∧ θ ≤ 5 * Real.pi / 12 :=
by sorry

end angle_inequality_l2744_274438


namespace sum_of_squares_of_roots_l2744_274411

theorem sum_of_squares_of_roots (r₁ r₂ : ℝ) : 
  r₁^2 - 10*r₁ + 9 = 0 →
  r₂^2 - 10*r₂ + 9 = 0 →
  (r₁ > 5 ∨ r₂ > 5) →
  r₁^2 + r₂^2 = 82 := by
sorry

end sum_of_squares_of_roots_l2744_274411


namespace discount_percentage_l2744_274439

def coffee_cost : ℝ := 6
def cheesecake_cost : ℝ := 10
def discounted_price : ℝ := 12

theorem discount_percentage : 
  (1 - discounted_price / (coffee_cost + cheesecake_cost)) * 100 = 25 := by
  sorry

end discount_percentage_l2744_274439


namespace reappearance_line_l2744_274429

def letter_cycle : List Char := ['B', 'K', 'I', 'G', 'N', 'O']
def digit_cycle : List Nat := [3, 0, 7, 2, 0]

theorem reappearance_line : 
  Nat.lcm (List.length letter_cycle) (List.length digit_cycle) = 30 := by
  sorry

end reappearance_line_l2744_274429


namespace even_function_condition_l2744_274412

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)

-- State the theorem
theorem even_function_condition (a : ℝ) : 
  (∀ x, f a x = f a (-x)) → a = -1 := by
  sorry

end even_function_condition_l2744_274412


namespace shooting_probabilities_l2744_274428

/-- Two shooters independently shoot at a target -/
structure ShootingScenario where
  /-- Probability of shooter A hitting the target -/
  prob_A : ℝ
  /-- Probability of shooter B hitting the target -/
  prob_B : ℝ
  /-- Assumption that probabilities are between 0 and 1 -/
  h_prob_A : 0 ≤ prob_A ∧ prob_A ≤ 1
  h_prob_B : 0 ≤ prob_B ∧ prob_B ≤ 1

/-- The probability that the target is hit in one shooting attempt -/
def prob_hit (s : ShootingScenario) : ℝ :=
  s.prob_A + s.prob_B - s.prob_A * s.prob_B

/-- The probability that the target is hit exactly by shooter A -/
def prob_hit_A (s : ShootingScenario) : ℝ :=
  s.prob_A * (1 - s.prob_B)

theorem shooting_probabilities (s : ShootingScenario) 
  (h_A : s.prob_A = 0.95) (h_B : s.prob_B = 0.9) : 
  prob_hit s = 0.995 ∧ prob_hit_A s = 0.095 := by
  sorry

#eval prob_hit ⟨0.95, 0.9, by norm_num, by norm_num⟩
#eval prob_hit_A ⟨0.95, 0.9, by norm_num, by norm_num⟩

end shooting_probabilities_l2744_274428


namespace arithmetic_progression_rth_term_l2744_274437

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 5 * n + 4 * n^3

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = 12 * r^2 - 12 * r + 9 := by
  sorry

end arithmetic_progression_rth_term_l2744_274437


namespace monday_sales_calculation_l2744_274468

def total_stock : ℕ := 1200
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 665/1000

theorem monday_sales_calculation :
  ∃ (monday_sales : ℕ),
    monday_sales = total_stock - 
      (tuesday_sales + wednesday_sales + thursday_sales + friday_sales) - 
      (unsold_percentage * total_stock).num :=
by sorry

end monday_sales_calculation_l2744_274468


namespace sin_210_degrees_l2744_274419

theorem sin_210_degrees : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_degrees_l2744_274419


namespace min_distance_to_line_l2744_274456

/-- The minimum distance from the origin to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
  ∀ (x y : ℝ), x + y - 4 = 0 → Real.sqrt (x^2 + y^2) ≥ d :=
by
  sorry

end min_distance_to_line_l2744_274456


namespace days_at_sisters_house_proof_l2744_274473

def vacation_duration : ℕ := 3 * 7

def known_days : ℕ := 1 + 5 + 1 + 5 + 1 + 1 + 1 + 1

def days_at_sisters_house : ℕ := vacation_duration - known_days

theorem days_at_sisters_house_proof :
  days_at_sisters_house = 5 := by
  sorry

end days_at_sisters_house_proof_l2744_274473


namespace harrys_morning_routine_time_l2744_274462

def morning_routine (coffee_bagel_time : ℕ) (reading_eating_factor : ℕ) : ℕ :=
  coffee_bagel_time + reading_eating_factor * coffee_bagel_time

theorem harrys_morning_routine_time :
  morning_routine 15 2 = 45 :=
by sorry

end harrys_morning_routine_time_l2744_274462


namespace equation_satisfied_at_x_equals_4_l2744_274435

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem equation_satisfied_at_x_equals_4 :
  2 * (f 4) - 19 = f (4 - 4) :=
by sorry

end equation_satisfied_at_x_equals_4_l2744_274435


namespace parabola_line_slope_l2744_274400

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - focus.1)

-- Define the condition for a point to be on both the line and the parabola
def intersection_point (k : ℝ) (x y : ℝ) : Prop :=
  parabola x y ∧ line_through_focus k x y

-- Define the ratio condition
def ratio_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = 16 * ((B.1 - focus.1)^2 + (B.2 - focus.2)^2)

theorem parabola_line_slope (k : ℝ) (A B : ℝ × ℝ) :
  intersection_point k A.1 A.2 →
  intersection_point k B.1 B.2 →
  A ≠ B →
  ratio_condition A B →
  k = 4/3 ∨ k = -4/3 :=
sorry

end parabola_line_slope_l2744_274400


namespace city_birth_rate_l2744_274489

/-- Represents the birth rate problem in a city --/
theorem city_birth_rate 
  (death_rate : ℕ) 
  (net_increase : ℕ) 
  (intervals_per_day : ℕ) 
  (h1 : death_rate = 3)
  (h2 : net_increase = 129600)
  (h3 : intervals_per_day = 43200) :
  ∃ (birth_rate : ℕ), 
    birth_rate = 6 ∧ 
    (birth_rate - death_rate) * intervals_per_day = net_increase :=
by sorry

end city_birth_rate_l2744_274489


namespace stratified_sampling_sample_size_l2744_274447

theorem stratified_sampling_sample_size 
  (ratio_old middle_aged young : ℕ) 
  (selected_middle_aged : ℕ) 
  (h1 : ratio_old = 4 ∧ middle_aged = 1 ∧ young = 5)
  (h2 : selected_middle_aged = 10) : 
  (selected_middle_aged : ℚ) / middle_aged * (ratio_old + middle_aged + young) = 100 := by
sorry

end stratified_sampling_sample_size_l2744_274447


namespace nathaniel_best_friends_l2744_274492

/-- Given that Nathaniel has 37 tickets initially, gives 5 tickets to each best friend,
    and ends up with 2 tickets, prove that he has 7 best friends. -/
theorem nathaniel_best_friends :
  let initial_tickets : ℕ := 37
  let tickets_per_friend : ℕ := 5
  let remaining_tickets : ℕ := 2
  let best_friends : ℕ := (initial_tickets - remaining_tickets) / tickets_per_friend
  best_friends = 7 := by
sorry


end nathaniel_best_friends_l2744_274492


namespace function_not_in_first_quadrant_l2744_274442

-- Define the linear function
def f (x : ℝ) : ℝ := -3 * x - 2

-- Theorem: The function f does not pass through the first quadrant
theorem function_not_in_first_quadrant :
  ∀ x y : ℝ, f x = y → ¬(x > 0 ∧ y > 0) := by
  sorry

end function_not_in_first_quadrant_l2744_274442


namespace xyz_product_l2744_274495

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5) (eq2 : y + 1/z = 2) (eq3 : z + 2/x = 10/3) :
  x * y * z = (21 + Real.sqrt 433) / 2 := by
  sorry

end xyz_product_l2744_274495


namespace det_A_formula_l2744_274430

theorem det_A_formula (n : ℕ) (h : n > 2) :
  let φ : ℝ := 2 * Real.pi / n
  let A : Matrix (Fin n) (Fin n) ℝ := λ i j =>
    if i = j then 1 + Real.cos (2 * φ * j) else Real.cos (φ * (i + j))
  Matrix.det A = -n^2 / 4 + 1 := by
  sorry

end det_A_formula_l2744_274430


namespace oh_squared_value_l2744_274416

/-- Given a triangle ABC with circumcenter O, orthocenter H, and circumradius R -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  O : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem states that for a triangle with R = 10 and a^2 + b^2 + c^2 = 50, OH^2 = 850 -/
theorem oh_squared_value (t : Triangle) 
  (h1 : t.R = 10) 
  (h2 : t.a^2 + t.b^2 + t.c^2 = 50) : 
  (t.O.1 - t.H.1)^2 + (t.O.2 - t.H.2)^2 = 850 := by
  sorry

end oh_squared_value_l2744_274416


namespace intersection_equals_universal_set_l2744_274464

theorem intersection_equals_universal_set {α : Type*} (S A B : Set α) 
  (h_universal : ∀ x, x ∈ S) 
  (h_intersection : A ∩ B = S) : 
  A = S ∧ B = S := by
  sorry

end intersection_equals_universal_set_l2744_274464


namespace cuboid_inequality_l2744_274465

theorem cuboid_inequality (x y z : ℝ) (hxy : x < y) (hyz : y < z)
  (p : ℝ) (hp : p = 4 * (x + y + z))
  (s : ℝ) (hs : s = 2 * (x*y + y*z + z*x))
  (d : ℝ) (hd : d = Real.sqrt (x^2 + y^2 + z^2)) :
  x < (1/3) * ((1/4) * p - Real.sqrt (d^2 - (1/2) * s)) ∧
  z > (1/3) * ((1/4) * p + Real.sqrt (d^2 - (1/2) * s)) := by
sorry

end cuboid_inequality_l2744_274465


namespace quadratic_integer_values_iff_coefficients_integer_l2744_274467

theorem quadratic_integer_values_iff_coefficients_integer (a b c : ℚ) :
  (∀ x : ℤ, ∃ n : ℤ, a * x^2 + b * x + c = n) ↔
  (∃ k : ℤ, 2 * a = k) ∧ (∃ m : ℤ, a + b = m) ∧ (∃ p : ℤ, c = p) :=
by sorry

end quadratic_integer_values_iff_coefficients_integer_l2744_274467


namespace max_daily_sales_l2744_274406

def P (t : ℕ) : ℝ :=
  if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else if 1 ≤ t ∧ t ≤ 24 then t + 20
  else 0

def Q (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 30 then -t + 40
  else 0

def y (t : ℕ) : ℝ := P t * Q t

theorem max_daily_sales :
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → y t ≤ 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 1125 → t = 25) :=
by sorry

end max_daily_sales_l2744_274406


namespace sin_sixty_degrees_l2744_274457

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_sixty_degrees_l2744_274457


namespace pyramid_properties_l2744_274402

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  base_area : ℝ
  cross_section_area1 : ℝ
  cross_section_area2 : ℝ
  cross_section_distance : ℝ

/-- Calculates the distance of the larger cross section from the apex -/
def larger_cross_section_distance (p : RightOctagonalPyramid) : ℝ := sorry

/-- Calculates the total height of the pyramid -/
def total_height (p : RightOctagonalPyramid) : ℝ := sorry

/-- Theorem stating the properties of the specific pyramid -/
theorem pyramid_properties (p : RightOctagonalPyramid) 
  (h1 : p.base_area = 1200)
  (h2 : p.cross_section_area1 = 300 * Real.sqrt 2)
  (h3 : p.cross_section_area2 = 675 * Real.sqrt 2)
  (h4 : p.cross_section_distance = 10) :
  larger_cross_section_distance p = 30 ∧ total_height p = 40 := by sorry

end pyramid_properties_l2744_274402


namespace same_remainder_divisor_l2744_274403

theorem same_remainder_divisor : ∃ (d : ℕ), d > 0 ∧ 
  ∀ (k : ℕ), k > d → 
  (∃ (r₁ r₂ r₃ : ℕ), 
    480608 = k * r₁ + d ∧
    508811 = k * r₂ + d ∧
    723217 = k * r₃ + d) → False :=
by
  -- The proof would go here
  sorry

end same_remainder_divisor_l2744_274403
