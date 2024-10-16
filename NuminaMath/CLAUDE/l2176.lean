import Mathlib

namespace NUMINAMATH_CALUDE_wind_velocity_problem_l2176_217605

/-- Represents the relationship between pressure, area, and wind velocity -/
def pressure_relation (k : ℝ) (A V : ℝ) : ℝ := k * A * V^3

theorem wind_velocity_problem (k : ℝ) :
  let A₁ : ℝ := 1
  let V₁ : ℝ := 10
  let P₁ : ℝ := 1
  let A₂ : ℝ := 1
  let P₂ : ℝ := 64
  pressure_relation k A₁ V₁ = P₁ →
  pressure_relation k A₂ 40 = P₂ :=
by
  sorry

#check wind_velocity_problem

end NUMINAMATH_CALUDE_wind_velocity_problem_l2176_217605


namespace NUMINAMATH_CALUDE_deer_per_hunting_wolf_l2176_217602

theorem deer_per_hunting_wolf (hunting_wolves : ℕ) (additional_wolves : ℕ) 
  (meat_per_wolf_per_day : ℕ) (days_between_hunts : ℕ) (meat_per_deer : ℕ) : 
  hunting_wolves = 4 →
  additional_wolves = 16 →
  meat_per_wolf_per_day = 8 →
  days_between_hunts = 5 →
  meat_per_deer = 200 →
  (hunting_wolves + additional_wolves) * meat_per_wolf_per_day * days_between_hunts / 
  (meat_per_deer * hunting_wolves) = 1 := by
sorry

end NUMINAMATH_CALUDE_deer_per_hunting_wolf_l2176_217602


namespace NUMINAMATH_CALUDE_constant_remainder_iff_a_eq_neg_35_l2176_217675

/-- The dividend polynomial -/
def dividend (a : ℚ) (x : ℚ) : ℚ := 10 * x^3 - 7 * x^2 + a * x + 10

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 2 * x^2 - 5 * x + 2

/-- The remainder when dividend is divided by divisor -/
def remainder (a : ℚ) (x : ℚ) : ℚ := dividend a x - divisor x * (5 * x + 15/2)

theorem constant_remainder_iff_a_eq_neg_35 :
  (∃ (c : ℚ), ∀ (x : ℚ), remainder a x = c) ↔ a = -35 := by sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_a_eq_neg_35_l2176_217675


namespace NUMINAMATH_CALUDE_pears_minus_apples_equals_two_l2176_217600

/-- Represents a bowl of fruit containing apples, pears, and bananas. -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- A bowl of fruit satisfying the given conditions. -/
def specialBowl : FruitBowl :=
  { apples := 0,  -- Placeholder value, will be constrained by theorem
    pears := 0,   -- Placeholder value, will be constrained by theorem
    bananas := 9 }

theorem pears_minus_apples_equals_two (bowl : FruitBowl) :
  bowl.apples + bowl.pears + bowl.bananas = 19 →
  bowl.bananas = 9 →
  bowl.bananas = bowl.pears + 3 →
  bowl.pears > bowl.apples →
  bowl.pears - bowl.apples = 2 := by
  sorry

#check pears_minus_apples_equals_two specialBowl

end NUMINAMATH_CALUDE_pears_minus_apples_equals_two_l2176_217600


namespace NUMINAMATH_CALUDE_dibromoalkane_formula_l2176_217632

/-- The mass fraction of bromine in a dibromoalkane -/
def bromine_mass_fraction : ℝ := 0.851

/-- The atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12

/-- The atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1

/-- The atomic mass of bromine in g/mol -/
def bromine_mass : ℝ := 80

/-- The general formula of a dibromoalkane is CₙH₂ₙBr₂ -/
def dibromoalkane_mass (n : ℕ) : ℝ :=
  n * carbon_mass + 2 * n * hydrogen_mass + 2 * bromine_mass

/-- Theorem: If the mass fraction of bromine in a dibromoalkane is 85.1%, then n = 2 -/
theorem dibromoalkane_formula :
  ∃ (n : ℕ), (2 * bromine_mass) / (dibromoalkane_mass n) = bromine_mass_fraction ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_dibromoalkane_formula_l2176_217632


namespace NUMINAMATH_CALUDE_quadratic_equation_prime_solutions_l2176_217679

theorem quadratic_equation_prime_solutions :
  ∀ (p q x₁ x₂ : ℤ),
    Prime p →
    Prime q →
    x₁^2 + p*x₁ + 3*q = 0 →
    x₂^2 + p*x₂ + 3*q = 0 →
    x₁ + x₂ = -p →
    x₁ * x₂ = 3*q →
    ((p = 7 ∧ q = 2 ∧ x₁ = -1 ∧ x₂ = -6) ∨
     (p = 5 ∧ q = 2 ∧ x₁ = -3 ∧ x₂ = -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_prime_solutions_l2176_217679


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2176_217669

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2176_217669


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2176_217620

theorem arithmetic_mean_problem :
  let numbers : Finset ℚ := {7/8, 9/10, 4/5, 17/20}
  17/20 ∈ numbers ∧ 
  9/10 ∈ numbers ∧ 
  4/5 ∈ numbers ∧
  (9/10 + 4/5) / 2 = 17/20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2176_217620


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l2176_217656

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7)) → 
  n ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l2176_217656


namespace NUMINAMATH_CALUDE_smallest_equal_packages_l2176_217612

theorem smallest_equal_packages (hamburger_pack : ℕ) (bun_pack : ℕ) : 
  hamburger_pack = 10 → bun_pack = 15 → 
  (∃ (h b : ℕ), h * hamburger_pack = b * bun_pack ∧ 
   ∀ (h' b' : ℕ), h' * hamburger_pack = b' * bun_pack → h ≤ h') → 
  (∃ (h : ℕ), h * hamburger_pack = 3 * hamburger_pack ∧ 
   ∃ (b : ℕ), b * bun_pack = 3 * hamburger_pack) :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_packages_l2176_217612


namespace NUMINAMATH_CALUDE_polygon_14_diagonals_interior_angles_sum_l2176_217608

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

theorem polygon_14_diagonals_interior_angles_sum :
  ∃ n : ℕ, num_diagonals n = 14 ∧ sum_interior_angles n = 900 :=
sorry

end NUMINAMATH_CALUDE_polygon_14_diagonals_interior_angles_sum_l2176_217608


namespace NUMINAMATH_CALUDE_complex_fourth_quadrant_m_range_l2176_217695

theorem complex_fourth_quadrant_m_range (m : ℝ) : 
  let z : ℂ := Complex.mk (m + 3) (m - 1)
  (0 < z.re ∧ z.im < 0) → -3 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_quadrant_m_range_l2176_217695


namespace NUMINAMATH_CALUDE_p_twelve_equals_neg_five_l2176_217633

/-- A quadratic function with specific properties -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

/-- Theorem stating that p(12) = -5 given certain conditions -/
theorem p_twelve_equals_neg_five 
  (d e f : ℝ) 
  (h1 : ∀ x, p d e f (3.5 + x) = p d e f (3.5 - x)) -- axis of symmetry at x = 3.5
  (h2 : p d e f (-5) = -5) -- p(-5) = -5
  : p d e f 12 = -5 := by
  sorry

end NUMINAMATH_CALUDE_p_twelve_equals_neg_five_l2176_217633


namespace NUMINAMATH_CALUDE_planet_coloring_l2176_217696

/-- Given 3 people coloring planets with 24 total colors, prove each person uses 8 colors --/
theorem planet_coloring (total_colors : ℕ) (num_people : ℕ) (h1 : total_colors = 24) (h2 : num_people = 3) :
  total_colors / num_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_planet_coloring_l2176_217696


namespace NUMINAMATH_CALUDE_oreo_count_l2176_217643

/-- The number of Oreos James has -/
def james_oreos : ℕ := 43

/-- The number of Oreos Jordan has -/
def jordan_oreos : ℕ := (james_oreos - 7) / 4

/-- The total number of Oreos between James and Jordan -/
def total_oreos : ℕ := james_oreos + jordan_oreos

theorem oreo_count : total_oreos = 52 := by
  sorry

end NUMINAMATH_CALUDE_oreo_count_l2176_217643


namespace NUMINAMATH_CALUDE_intersection_sum_l2176_217651

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 2
def g (x y : ℝ) : Prop := x + 2*y = 2

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
  f x₁ = y₁ ∧ g x₁ y₁ ∧
  f x₂ = y₂ ∧ g x₂ y₂ ∧
  f x₃ = y₃ ∧ g x₃ y₃

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    f x₁ = y₁ ∧ g x₁ y₁ ∧
    f x₂ = y₂ ∧ g x₂ y₂ ∧
    f x₃ = y₃ ∧ g x₃ y₃ ∧
    x₁ + x₂ + x₃ = 4 ∧
    y₁ + y₂ + y₃ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2176_217651


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_11_plus_1_divisible_by_13_l2176_217640

theorem smallest_number_divisible_by_11_plus_1_divisible_by_13 :
  ∃ n : ℕ, n = 77 ∧
  (∀ m : ℕ, m < n → ¬(11 ∣ m ∧ 13 ∣ (m + 1))) ∧
  11 ∣ n ∧ 13 ∣ (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_11_plus_1_divisible_by_13_l2176_217640


namespace NUMINAMATH_CALUDE_eunji_class_size_l2176_217606

/-- The number of students in Eunji's class -/
def class_size : ℕ := 24

/-- The number of lines the students stand in -/
def num_lines : ℕ := 3

/-- Eunji's position from the front of her row -/
def position_from_front : ℕ := 3

/-- Eunji's position from the back of her row -/
def position_from_back : ℕ := 6

/-- Theorem stating the number of students in Eunji's class -/
theorem eunji_class_size :
  class_size = num_lines * (position_from_front + position_from_back - 1) :=
by sorry

end NUMINAMATH_CALUDE_eunji_class_size_l2176_217606


namespace NUMINAMATH_CALUDE_expected_ones_three_dice_l2176_217652

-- Define a standard die
def standardDie : Finset Nat := Finset.range 6

-- Define the probability of rolling a 1 on a single die
def probOne : ℚ := 1 / 6

-- Define the probability of not rolling a 1 on a single die
def probNotOne : ℚ := 1 - probOne

-- Define the number of dice
def numDice : Nat := 3

-- Define the expected value function for discrete random variables
def expectedValue (outcomes : Finset Nat) (prob : Nat → ℚ) : ℚ :=
  Finset.sum outcomes (λ x => x * prob x)

-- Statement of the theorem
theorem expected_ones_three_dice :
  expectedValue (Finset.range (numDice + 1)) (λ k =>
    (numDice.choose k : ℚ) * probOne ^ k * probNotOne ^ (numDice - k)) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_expected_ones_three_dice_l2176_217652


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l2176_217618

-- Define the group
def total_boys : ℕ := 5
def total_girls : ℕ := 3

-- Define the events
def exactly_one_boy (selected_boys : ℕ) : Prop := selected_boys = 1
def exactly_two_girls (selected_girls : ℕ) : Prop := selected_girls = 2

-- Define the sample space
def sample_space : Set (ℕ × ℕ) :=
  {pair | pair.1 + pair.2 = 2 ∧ pair.1 ≤ total_boys ∧ pair.2 ≤ total_girls}

-- Theorem to prove
theorem events_mutually_exclusive_not_contradictory :
  (∃ (pair : ℕ × ℕ), pair ∈ sample_space ∧ exactly_one_boy pair.1 ∧ ¬exactly_two_girls pair.2) ∧
  (∃ (pair : ℕ × ℕ), pair ∈ sample_space ∧ ¬exactly_one_boy pair.1 ∧ exactly_two_girls pair.2) ∧
  (¬∃ (pair : ℕ × ℕ), pair ∈ sample_space ∧ exactly_one_boy pair.1 ∧ exactly_two_girls pair.2) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l2176_217618


namespace NUMINAMATH_CALUDE_quadratic_equation_at_most_one_solution_l2176_217615

theorem quadratic_equation_at_most_one_solution (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a ≥ 9/8 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_at_most_one_solution_l2176_217615


namespace NUMINAMATH_CALUDE_fraction_denominator_l2176_217692

theorem fraction_denominator (x : ℚ) : 
  (525 : ℚ) / x = (21 : ℚ) / 40 →
  (∃ n : ℕ, n ≥ 81 ∧ (525 : ℚ) / x - ((525 : ℚ) / x).floor = 5 / (10 ^ n)) →
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_fraction_denominator_l2176_217692


namespace NUMINAMATH_CALUDE_age_sum_proof_l2176_217637

theorem age_sum_proof (asaf_age : ℕ) (alexander_age : ℕ) 
  (asaf_pencils : ℕ) (alexander_pencils : ℕ) : 
  asaf_age = 50 →
  asaf_age - alexander_age = asaf_pencils / 2 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 →
  asaf_age + alexander_age = 60 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l2176_217637


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2176_217650

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ :=
  if a ≥ b then b^2 else 2*a - b

-- Theorem statements
theorem problem_1 : triangle (-4) (-5) = 25 := by sorry

theorem problem_2 : triangle (triangle (-3) 2) (-9) = 81 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2176_217650


namespace NUMINAMATH_CALUDE_lollipops_per_boy_l2176_217694

theorem lollipops_per_boy (total_candies : ℕ) (total_children : ℕ) 
  (h1 : total_candies = 90)
  (h2 : total_children = 40)
  (h3 : ∃ (num_lollipops : ℕ), num_lollipops = total_candies / 3)
  (h4 : ∃ (num_candy_canes : ℕ), num_candy_canes = total_candies - total_candies / 3)
  (h5 : ∃ (num_girls : ℕ), num_girls = (total_candies - total_candies / 3) / 2)
  (h6 : ∃ (num_boys : ℕ), num_boys = total_children - (total_candies - total_candies / 3) / 2) :
  (total_candies / 3) / (total_children - (total_candies - total_candies / 3) / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_lollipops_per_boy_l2176_217694


namespace NUMINAMATH_CALUDE_fair_coin_toss_is_fair_l2176_217689

-- Define a fair coin
def fair_coin (outcome : Bool) : ℝ :=
  if outcome then 0.5 else 0.5

-- Define fairness of a decision method
def is_fair (decision_method : Bool → ℝ) : Prop :=
  decision_method true = decision_method false

-- Theorem statement
theorem fair_coin_toss_is_fair :
  is_fair fair_coin :=
sorry

end NUMINAMATH_CALUDE_fair_coin_toss_is_fair_l2176_217689


namespace NUMINAMATH_CALUDE_weight_difference_l2176_217674

/-- Given that Antoinette and Rupert have a combined weight of 98 kg,
    and Antoinette weighs 63 kg, prove that Antoinette weighs 7 kg less
    than twice Rupert's weight. -/
theorem weight_difference (antoinette_weight rupert_weight : ℝ) : 
  antoinette_weight = 63 →
  antoinette_weight + rupert_weight = 98 →
  2 * rupert_weight - antoinette_weight = 7 :=
by sorry

end NUMINAMATH_CALUDE_weight_difference_l2176_217674


namespace NUMINAMATH_CALUDE_xiaoming_calculation_correction_l2176_217603

theorem xiaoming_calculation_correction 
  (A a b c : ℝ) 
  (h : A + 2 * (a * b + 2 * b * c - 4 * a * c) = 3 * a * b - 2 * a * c + 5 * b * c) : 
  A - 2 * (a * b + 2 * b * c - 4 * a * c) = -a * b + 14 * a * c - 3 * b * c := by
sorry

end NUMINAMATH_CALUDE_xiaoming_calculation_correction_l2176_217603


namespace NUMINAMATH_CALUDE_function_properties_l2176_217617

/-- The given function f(x) -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + 1

/-- Theorem stating the properties of the function f -/
theorem function_properties (A ω φ : ℝ) (h1 : A > 0) (h2 : ω > 0) (h3 : -π/2 ≤ φ ∧ φ ≤ π/2) :
  (∀ x, f A ω φ x = f A ω φ (2*π/3 - x)) → -- Symmetry about x = π/3
  (∃ x, f A ω φ x = 3) → -- Maximum value is 3
  (∀ x, f A ω φ x = f A ω φ (x + π)) → -- Distance between highest points is π
  (∃ θ, f A ω φ (θ/2 + π/3) = 7/5) →
  (∀ x, f A ω φ x = f A ω φ (x + π)) ∧ -- Smallest positive period is π
  (∀ x, f A ω φ x = 2 * Real.sin (2*x - π/6) + 1) ∧ -- Analytical expression
  (∀ θ, f A ω φ (θ/2 + π/3) = 7/5 → Real.sin θ = 2*Real.sqrt 6/5 ∨ Real.sin θ = -2*Real.sqrt 6/5) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2176_217617


namespace NUMINAMATH_CALUDE_not_perfect_square_l2176_217634

theorem not_perfect_square (m n : ℕ) : ¬ ∃ k : ℕ, 1 + 3^m + 3^n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2176_217634


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l2176_217683

theorem triangle_angle_inequality (A B C α : Real) : 
  A + B + C = π →
  A > 0 → B > 0 → C > 0 →
  α = min (2 * A - B) (min (3 * B - 2 * C) (π / 2 - A)) →
  α ≤ 2 * π / 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l2176_217683


namespace NUMINAMATH_CALUDE_cranberry_harvest_percentage_l2176_217616

/-- Given the initial number of cranberries, the number eaten by elk, and the number left,
    prove that the percentage of cranberries harvested by humans is 40%. -/
theorem cranberry_harvest_percentage
  (total : ℕ)
  (eaten_by_elk : ℕ)
  (left : ℕ)
  (h1 : total = 60000)
  (h2 : eaten_by_elk = 20000)
  (h3 : left = 16000) :
  (total - eaten_by_elk - left) / total * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cranberry_harvest_percentage_l2176_217616


namespace NUMINAMATH_CALUDE_base_6_arithmetic_l2176_217686

/-- Convert a base 6 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (λ acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a base 10 number to base 6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Theorem: 1254₆ - 432₆ + 221₆ = 1043₆ in base 6 --/
theorem base_6_arithmetic :
  to_base_6 (to_base_10 [4, 5, 2, 1] - to_base_10 [2, 3, 4] + to_base_10 [1, 2, 2]) = [3, 4, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_6_arithmetic_l2176_217686


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2176_217657

theorem fraction_equation_solution (x : ℚ) :
  2/5 - 1/4 = 1/x → x = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2176_217657


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l2176_217630

theorem smallest_common_multiple_of_8_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, 8 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 8 ∣ n ∧ 6 ∣ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l2176_217630


namespace NUMINAMATH_CALUDE_sons_age_l2176_217685

theorem sons_age (son father : ℕ) 
  (h1 : son = (father / 4) - 1)
  (h2 : father = 5 * son - 5) : 
  son = 9 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2176_217685


namespace NUMINAMATH_CALUDE_min_value_trig_function_l2176_217629

theorem min_value_trig_function (x : ℝ) : 
  Real.sin x ^ 4 + Real.cos x ^ 4 + (1 / Real.cos x) ^ 4 + (1 / Real.sin x) ^ 4 ≥ 8.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_function_l2176_217629


namespace NUMINAMATH_CALUDE_percentage_relation_l2176_217601

/-- Given three real numbers A, B, and C, where A is 8% of C and 50% of B,
    prove that B is 16% of C. -/
theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.08 * C) 
  (h2 : A = 0.5 * B) : 
  B = 0.16 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l2176_217601


namespace NUMINAMATH_CALUDE_charlie_cleaning_time_l2176_217648

theorem charlie_cleaning_time (alice_time bob_time charlie_time : ℚ) : 
  alice_time = 30 →
  bob_time = (3 / 4) * alice_time →
  charlie_time = (1 / 3) * bob_time →
  charlie_time = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_charlie_cleaning_time_l2176_217648


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l2176_217626

-- Define the curve function
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_point_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  f' x₀ = 5 ∧ f x₀ = y₀ ∧ y₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l2176_217626


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2176_217668

theorem shaded_area_percentage (side_length : ℝ) (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℝ) :
  side_length = 6 ∧
  rect1_width = 2 ∧ rect1_height = 2 ∧
  rect2_width = 4 ∧ rect2_height = 1 ∧
  rect3_width = 6 ∧ rect3_height = 1 →
  (rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height) / (side_length * side_length) = 22 / 36 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2176_217668


namespace NUMINAMATH_CALUDE_quarters_count_l2176_217644

/-- Calculates the number of quarters in a jar given the following conditions:
  * The jar contains 123 pennies, 85 nickels, 35 dimes, and an unknown number of quarters.
  * The total cost of ice cream for 5 family members is $15.
  * After spending on ice cream, 48 cents remain. -/
def quarters_in_jar (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (ice_cream_cost : ℚ) (remaining_cents : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of quarters in the jar is 26. -/
theorem quarters_count : quarters_in_jar 123 85 35 15 48 = 26 := by
  sorry

end NUMINAMATH_CALUDE_quarters_count_l2176_217644


namespace NUMINAMATH_CALUDE_ratio_problem_l2176_217635

theorem ratio_problem (a b : ℕ) (h1 : a = 55) (h2 : a = 5 * b) : b = 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2176_217635


namespace NUMINAMATH_CALUDE_not_perfect_square_l2176_217655

theorem not_perfect_square (n : ℕ) : ¬ ∃ (k : ℤ), (3 : ℤ)^n + 2 * 17^n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2176_217655


namespace NUMINAMATH_CALUDE_min_moves_to_equalize_l2176_217680

/-- Represents a stack of coins -/
structure CoinStack :=
  (coins : ℕ)

/-- Represents the state of all coin stacks -/
structure CoinStacks :=
  (stacks : Fin 4 → CoinStack)

/-- Represents a move that adds one coin to three different stacks -/
structure Move :=
  (targets : Fin 3 → Fin 4)
  (different : targets 0 ≠ targets 1 ∧ targets 0 ≠ targets 2 ∧ targets 1 ≠ targets 2)

/-- The initial state of the coin stacks -/
def initial_stacks : CoinStacks :=
  CoinStacks.mk (fun i => match i with
    | 0 => CoinStack.mk 9
    | 1 => CoinStack.mk 7
    | 2 => CoinStack.mk 5
    | 3 => CoinStack.mk 10)

/-- Applies a move to a given state of coin stacks -/
def apply_move (stacks : CoinStacks) (move : Move) : CoinStacks :=
  sorry

/-- Checks if all stacks have an equal number of coins -/
def all_equal (stacks : CoinStacks) : Prop :=
  sorry

/-- The main theorem to prove -/
theorem min_moves_to_equalize :
  ∃ (moves : List Move),
    moves.length = 11 ∧
    all_equal (moves.foldl apply_move initial_stacks) ∧
    ∀ (other_moves : List Move),
      all_equal (other_moves.foldl apply_move initial_stacks) →
      other_moves.length ≥ 11 :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_equalize_l2176_217680


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_111_l2176_217667

theorem smallest_six_digit_divisible_by_111 :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 → n ≥ 100011 :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_111_l2176_217667


namespace NUMINAMATH_CALUDE_transmission_time_is_256_seconds_l2176_217611

/-- Represents the number of blocks to be sent -/
def num_blocks : ℕ := 40

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 1024

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 160

/-- Theorem stating that the transmission time is 256 seconds -/
theorem transmission_time_is_256_seconds :
  (num_blocks * chunks_per_block) / transmission_rate = 256 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_is_256_seconds_l2176_217611


namespace NUMINAMATH_CALUDE_range_of_a_l2176_217653

-- Define the function f
def f (x : ℝ) : ℝ := 3*x + 2*x^3

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 2, f x = 3*x + 2*x^3) →
  (f (a - 1) + f (1 - 2*a) < 0) →
  a ∈ Set.Ioo 0 (3/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2176_217653


namespace NUMINAMATH_CALUDE_unanswered_test_completion_ways_l2176_217698

/-- Represents a multiple choice test. -/
structure MCTest where
  num_questions : Nat
  choices_per_question : Nat

/-- Defines an unanswered test. -/
def unanswered_test (test : MCTest) : Nat := 1

/-- Theorem stating that for a test with 4 questions and 5 choices per question,
    there is only one way to complete it if all questions are unanswered. -/
theorem unanswered_test_completion_ways :
  let test : MCTest := { num_questions := 4, choices_per_question := 5 }
  unanswered_test test = 1 := by
  sorry

end NUMINAMATH_CALUDE_unanswered_test_completion_ways_l2176_217698


namespace NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l2176_217687

def calculate_koolaid_percentage (initial_powder : ℚ) (initial_water : ℚ) (evaporated_water : ℚ) (water_multiplier : ℚ) : ℚ :=
  let remaining_water := initial_water - evaporated_water
  let final_water := remaining_water * water_multiplier
  let total_liquid := initial_powder + final_water
  (initial_powder / total_liquid) * 100

theorem koolaid_percentage_is_four_percent :
  calculate_koolaid_percentage 2 16 4 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l2176_217687


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_terms_l2176_217672

def arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, i < j → j ≤ n → a j - a i = (j - i : ℝ) * (a 2 - a 1)

theorem arithmetic_sequence_n_terms
  (a : ℕ → ℝ) (n : ℕ)
  (h1 : arithmetic_sequence a n)
  (h2 : a 1 + a 2 + a 3 = 20)
  (h3 : a (n-2) + a (n-1) + a n = 130)
  (h4 : (Finset.range n).sum a = 200) :
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_terms_l2176_217672


namespace NUMINAMATH_CALUDE_b_visited_zhougong_l2176_217645

-- Define the celebrities
inductive Celebrity
| A
| B
| C

-- Define the places
inductive Place
| ZhougongTemple
| FamenTemple
| Wuzhangyuan

-- Define a function to represent whether a celebrity visited a place
def visited : Celebrity → Place → Prop := sorry

-- A visited more places than B
axiom a_visited_more : ∃ (p : Place), visited Celebrity.A p ∧ ¬visited Celebrity.B p

-- A did not visit Famen Temple
axiom a_not_famen : ¬visited Celebrity.A Place.FamenTemple

-- B did not visit Wuzhangyuan
axiom b_not_wuzhangyuan : ¬visited Celebrity.B Place.Wuzhangyuan

-- The three celebrities visited the same place
axiom same_place : ∃ (p : Place), visited Celebrity.A p ∧ visited Celebrity.B p ∧ visited Celebrity.C p

-- Theorem to prove
theorem b_visited_zhougong : visited Celebrity.B Place.ZhougongTemple := by sorry

end NUMINAMATH_CALUDE_b_visited_zhougong_l2176_217645


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l2176_217671

/-- Represents a digit (1-9) -/
def Digit := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- Converts a two-digit number to its decimal representation -/
def twoDigitToNum (a b : Digit) : ℕ := 10 * a.val + b.val

/-- Converts a three-digit number with all digits the same to its decimal representation -/
def threeDigitSameToNum (c : Digit) : ℕ := 100 * c.val + 10 * c.val + c.val

theorem cryptarithm_solution :
  ∃! (a b c : Digit),
    a.val ≠ b.val ∧ b.val ≠ c.val ∧ a.val ≠ c.val ∧
    twoDigitToNum a b + a.val * threeDigitSameToNum c = 247 ∧
    a.val = 2 ∧ b.val = 5 ∧ c.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l2176_217671


namespace NUMINAMATH_CALUDE_gcd_540_180_minus_2_l2176_217682

theorem gcd_540_180_minus_2 : Int.gcd 540 180 - 2 = 178 := by
  sorry

end NUMINAMATH_CALUDE_gcd_540_180_minus_2_l2176_217682


namespace NUMINAMATH_CALUDE_total_travel_time_l2176_217628

def luke_bus_time : ℕ := 70
def paula_bus_time : ℕ := (3 * luke_bus_time) / 5
def jane_train_time : ℕ := 120
def michael_cycle_time : ℕ := jane_train_time / 4

def luke_total_time : ℕ := luke_bus_time + 5 * luke_bus_time
def paula_total_time : ℕ := 2 * paula_bus_time
def jane_total_time : ℕ := jane_train_time + 2 * jane_train_time
def michael_total_time : ℕ := 2 * michael_cycle_time

theorem total_travel_time :
  luke_total_time + paula_total_time + jane_total_time + michael_total_time = 924 :=
by sorry

end NUMINAMATH_CALUDE_total_travel_time_l2176_217628


namespace NUMINAMATH_CALUDE_area_traced_by_rolling_triangle_l2176_217673

/-- The area traced out by rolling an equilateral triangle -/
theorem area_traced_by_rolling_triangle (side_length : ℝ) (h : side_length = 6) :
  let triangle_height : ℝ := side_length * Real.sqrt 3 / 2
  let arc_length : ℝ := π * side_length / 3
  let rectangle_area : ℝ := side_length * arc_length
  let quarter_circle_area : ℝ := π * side_length^2 / 4
  rectangle_area + quarter_circle_area = 21 * π := by
  sorry

#check area_traced_by_rolling_triangle

end NUMINAMATH_CALUDE_area_traced_by_rolling_triangle_l2176_217673


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l2176_217663

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 2, f x = max) ∧
    (∀ x ∈ Set.Icc 0 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 2, f x = min) ∧
    max = 5 ∧ min = -15 := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l2176_217663


namespace NUMINAMATH_CALUDE_brother_travel_distance_l2176_217684

theorem brother_travel_distance (total_time : ℝ) (speed_diff : ℝ) (distance_diff : ℝ) :
  total_time = 120 ∧ speed_diff = 4 ∧ distance_diff = 40 →
  ∃ (x y : ℝ),
    x = 20 ∧ y = 60 ∧
    total_time / x - total_time / y = speed_diff ∧
    y - x = distance_diff :=
by sorry

end NUMINAMATH_CALUDE_brother_travel_distance_l2176_217684


namespace NUMINAMATH_CALUDE_probability_small_area_is_two_thirds_l2176_217661

/-- A right triangle XYZ with vertices X=(0,8), Y=(0,0), Z=(10,0) -/
structure RightTriangle where
  X : ℝ × ℝ := (0, 8)
  Y : ℝ × ℝ := (0, 0)
  Z : ℝ × ℝ := (10, 0)

/-- The area of a triangle given three points -/
def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- The probability that a randomly chosen point Q in the interior of XYZ
    satisfies area(QYZ) < 1/3 * area(XYZ) -/
def probabilitySmallArea (t : RightTriangle) : ℝ := sorry

/-- Theorem: The probability that area(QYZ) < 1/3 * area(XYZ) is 2/3 -/
theorem probability_small_area_is_two_thirds (t : RightTriangle) :
  probabilitySmallArea t = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_small_area_is_two_thirds_l2176_217661


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2176_217662

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x ≠ 1 ∧ (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2176_217662


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2176_217641

theorem ratio_sum_problem (x y z b : ℚ) : 
  x / y = 4 / 5 →
  y / z = 5 / 6 →
  y = 15 * b - 5 →
  x + y + z = 90 →
  b = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2176_217641


namespace NUMINAMATH_CALUDE_johnny_savings_l2176_217697

/-- The amount Johnny saved in September -/
def september_savings : ℕ := 30

/-- The amount Johnny saved in October -/
def october_savings : ℕ := 49

/-- The amount Johnny saved in November -/
def november_savings : ℕ := 46

/-- The amount Johnny spent on a video game -/
def video_game_cost : ℕ := 58

/-- The amount Johnny has left after all transactions -/
def remaining_money : ℕ := 67

theorem johnny_savings : 
  september_savings + october_savings + november_savings - video_game_cost = remaining_money := by
  sorry

end NUMINAMATH_CALUDE_johnny_savings_l2176_217697


namespace NUMINAMATH_CALUDE_normal_symmetric_probability_l2176_217693

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- The cumulative distribution function for a normal random variable -/
noncomputable def normalCDF (ξ : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_symmetric_probability (ξ : NormalRV) (h : ξ.μ = 2) 
  (h_cdf : normalCDF ξ 4 = 0.8) : 
  normalCDF ξ 2 - normalCDF ξ 0 = 0.3 := by sorry

end NUMINAMATH_CALUDE_normal_symmetric_probability_l2176_217693


namespace NUMINAMATH_CALUDE_triangle_angle_identity_l2176_217607

theorem triangle_angle_identity (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_less_than_pi : α < Real.pi ∧ β < Real.pi ∧ γ < Real.pi) : 
  (Real.cos α) / (Real.sin β * Real.sin γ) + 
  (Real.cos β) / (Real.sin α * Real.sin γ) + 
  (Real.cos γ) / (Real.sin α * Real.sin β) = 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_angle_identity_l2176_217607


namespace NUMINAMATH_CALUDE_min_circle_property_l2176_217625

/-- Definition of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + y = -1

/-- Definition of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- Definition of the circle with minimal area -/
def minCircle (x y : ℝ) : Prop := x^2 + y^2 + (6/5)*x + (3/5)*y + 1 = 0

/-- Theorem stating that the minCircle passes through the intersection points of circle1 and circle2 and has the minimum area -/
theorem min_circle_property :
  ∀ (x y : ℝ), 
    (circle1 x y ∧ circle2 x y → minCircle x y) ∧
    (∀ (a b c : ℝ), (∀ (u v : ℝ), circle1 u v ∧ circle2 u v → x^2 + y^2 + 2*a*x + 2*b*y + c = 0) →
      (x^2 + y^2 + (6/5)*x + (3/5)*y + 1)^2 ≤ (x^2 + y^2 + 2*a*x + 2*b*y + c)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_circle_property_l2176_217625


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2176_217678

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given condition: a_2 + a_5 + a_8 = 6 -/
def GivenCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 5 + a 8 = 6

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : GivenCondition a) : a 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2176_217678


namespace NUMINAMATH_CALUDE_thirty_three_not_enrolled_l2176_217627

/-- Calculates the number of students not enrolled in either French or German --/
def students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) : ℕ :=
  total - (french + german - both)

/-- Theorem stating that 33 students are not enrolled in either French or German --/
theorem thirty_three_not_enrolled : 
  students_not_enrolled 87 41 22 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_thirty_three_not_enrolled_l2176_217627


namespace NUMINAMATH_CALUDE_difference_of_differences_l2176_217613

theorem difference_of_differences (a b c : ℤ) 
  (h1 : a - b = 2) 
  (h2 : b - c = -3) : 
  a - c = -1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_differences_l2176_217613


namespace NUMINAMATH_CALUDE_function_minimum_condition_l2176_217631

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem function_minimum_condition (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f a x₀ ≤ f a x) ∧ 
  (∀ x : ℝ, f a x ≥ 2 * a^2 - a - 1) ↔ 
  0 < a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_function_minimum_condition_l2176_217631


namespace NUMINAMATH_CALUDE_reception_friends_l2176_217609

def wedding_reception (total_attendees : ℕ) (bride_couples : ℕ) (groom_couples : ℕ) : Prop :=
  let family_members := 2 * (bride_couples + groom_couples)
  let friends := total_attendees - family_members
  friends = 100

theorem reception_friends :
  wedding_reception 180 20 20 := by
  sorry

end NUMINAMATH_CALUDE_reception_friends_l2176_217609


namespace NUMINAMATH_CALUDE_new_ratio_after_subtraction_l2176_217677

theorem new_ratio_after_subtraction :
  let a : ℚ := 72
  let b : ℚ := 192
  let subtrahend : ℚ := 24
  (a / b = 3 / 8) →
  ((a - subtrahend) / (b - subtrahend) = 1 / (7/2)) :=
by sorry

end NUMINAMATH_CALUDE_new_ratio_after_subtraction_l2176_217677


namespace NUMINAMATH_CALUDE_f_properties_l2176_217647

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_properties (a : ℝ) :
  (∀ x, a ≤ 0 → deriv (f a) x ≤ 0) ∧
  (a > 0 → ∀ x, x < Real.log (1/a) → deriv (f a) x < 0) ∧
  (a > 0 → ∀ x, x > Real.log (1/a) → deriv (f a) x > 0) ∧
  (a > 0 → ∀ x, f a x > 2 * Real.log a + 3/2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2176_217647


namespace NUMINAMATH_CALUDE_cuboid_color_is_blue_l2176_217610

/-- Represents a cube with colored faces -/
structure ColoredCube where
  red_faces : Fin 6
  blue_faces : Fin 6
  yellow_faces : Fin 6
  face_sum : red_faces + blue_faces + yellow_faces = 6

/-- Represents the arrangement of cubes in a photo -/
structure CubeArrangement where
  red_visible : Nat
  blue_visible : Nat
  yellow_visible : Nat
  total_visible : red_visible + blue_visible + yellow_visible = 8

/-- The set of four cubes -/
def cube_set : Finset ColoredCube := sorry

/-- The three different arrangements in the colored photos -/
def arrangements : Finset CubeArrangement := sorry

theorem cuboid_color_is_blue 
  (h1 : ∀ c ∈ cube_set, c.red_faces + c.blue_faces + c.yellow_faces = 6)
  (h2 : cube_set.card = 4)
  (h3 : ∀ a ∈ arrangements, a.red_visible + a.blue_visible + a.yellow_visible = 8)
  (h4 : arrangements.card = 3)
  (h5 : (arrangements.sum (λ a => a.red_visible)) = 8)
  (h6 : (arrangements.sum (λ a => a.blue_visible)) = 8)
  (h7 : (arrangements.sum (λ a => a.yellow_visible)) = 8)
  (h8 : ∃ a ∈ arrangements, a.red_visible = 2)
  (h9 : ∃ c ∈ cube_set, c.yellow_faces = 0) :
  ∀ c ∈ cube_set, c.blue_faces ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_cuboid_color_is_blue_l2176_217610


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l2176_217624

-- Problem 1
theorem simplify_expression (x y : ℝ) :
  5 * (2 * x^3 * y + 3 * x * y^2) - (6 * x * y^2 - 3 * x^3 * y) = 13 * x^3 * y + 9 * x * y^2 := by
  sorry

-- Problem 2
theorem evaluate_expression (a b : ℝ) (h1 : a + b = 9) (h2 : a * b = 20) :
  2/3 * (-15 * a + 3 * a * b) + 1/5 * (2 * a * b - 10 * a) - 4 * (a * b + 3 * b) = -140 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l2176_217624


namespace NUMINAMATH_CALUDE_fraction_addition_l2176_217619

theorem fraction_addition : (18 : ℚ) / 42 + 2 / 9 = 41 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2176_217619


namespace NUMINAMATH_CALUDE_M_bisects_AB_l2176_217639

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 2) / 3 = (y + 1) / 5 ∧ (x - 2) / 3 = (z - 3) / (-1)

/-- Point A where the line intersects the xoz plane -/
def point_A : ℝ × ℝ × ℝ := (2.6, 0, 2.8)

/-- Point B where the line intersects the xoy plane -/
def point_B : ℝ × ℝ × ℝ := (11, 14, 0)

/-- Point M that supposedly bisects AB -/
def point_M : ℝ × ℝ × ℝ := (6.8, 7, 1.4)

/-- Theorem stating that M bisects AB -/
theorem M_bisects_AB :
  line_equation point_A.1 point_A.2.1 point_A.2.2 ∧
  line_equation point_B.1 point_B.2.1 point_B.2.2 ∧
  point_M.1 = (point_A.1 + point_B.1) / 2 ∧
  point_M.2.1 = (point_A.2.1 + point_B.2.1) / 2 ∧
  point_M.2.2 = (point_A.2.2 + point_B.2.2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_M_bisects_AB_l2176_217639


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2176_217638

theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) :
  a + b = (2, -8) →
  a - b = (-8, 16) →
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  cos_theta = -63/65 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2176_217638


namespace NUMINAMATH_CALUDE_second_person_age_l2176_217622

/-- Given a group of 7 people, if adding a 39-year-old increases the average age by 2,
    and adding another person decreases the average age by 1,
    then the age of the second person added is 15 years old. -/
theorem second_person_age (initial_group : Finset ℕ) 
  (initial_total_age : ℕ) (second_person_age : ℕ) :
  (initial_group.card = 7) →
  (initial_total_age / 7 + 2 = (initial_total_age + 39) / 8) →
  (initial_total_age / 7 - 1 = (initial_total_age + second_person_age) / 8) →
  second_person_age = 15 := by
  sorry


end NUMINAMATH_CALUDE_second_person_age_l2176_217622


namespace NUMINAMATH_CALUDE_correct_calculation_l2176_217649

theorem correct_calculation (x : ℤ) (h : x + 5 = 43) : 5 * x = 190 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2176_217649


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l2176_217614

theorem consecutive_integers_average (x : ℝ) : 
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 25 →
  ((x - 9) + (x + 1 - 8) + (x + 2 - 7) + (x + 3 - 6) + (x + 4 - 5) + (x + 5 - 4) + (x + 6 - 3) + (x + 7 - 2) + (x + 8 - 1) + (x + 9)) / 10 = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l2176_217614


namespace NUMINAMATH_CALUDE_third_year_sample_size_l2176_217604

/-- Calculates the number of students to be sampled from a specific grade in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (grade_population : ℕ) (total_sample_size : ℕ) : ℕ :=
  (grade_population * total_sample_size) / total_population

/-- Proves that the number of third-year students to be sampled is 21 -/
theorem third_year_sample_size :
  let total_population : ℕ := 3000
  let third_year_population : ℕ := 1050
  let total_sample_size : ℕ := 60
  stratified_sample_size total_population third_year_population total_sample_size = 21 := by
  sorry


end NUMINAMATH_CALUDE_third_year_sample_size_l2176_217604


namespace NUMINAMATH_CALUDE_g_composition_result_l2176_217670

-- Define the complex function g
noncomputable def g (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^3 else -z^3

-- State the theorem
theorem g_composition_result :
  g (g (g (g (1 + Complex.I)))) = -8192 - 45056 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_g_composition_result_l2176_217670


namespace NUMINAMATH_CALUDE_other_birds_percentage_l2176_217658

/-- Represents the composition of birds in the Goshawk-Eurasian Nature Reserve -/
structure BirdReserve where
  total : ℝ
  hawk_percent : ℝ
  paddyfield_warbler_percent_of_nonhawk : ℝ
  kingfisher_to_paddyfield_warbler_ratio : ℝ

/-- Theorem stating the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
theorem other_birds_percentage (reserve : BirdReserve) 
  (h1 : reserve.hawk_percent = 0.3)
  (h2 : reserve.paddyfield_warbler_percent_of_nonhawk = 0.4)
  (h3 : reserve.kingfisher_to_paddyfield_warbler_ratio = 0.25)
  (h4 : reserve.total > 0) :
  let hawk_count := reserve.hawk_percent * reserve.total
  let nonhawk_count := reserve.total - hawk_count
  let paddyfield_warbler_count := reserve.paddyfield_warbler_percent_of_nonhawk * nonhawk_count
  let kingfisher_count := reserve.kingfisher_to_paddyfield_warbler_ratio * paddyfield_warbler_count
  let other_count := reserve.total - (hawk_count + paddyfield_warbler_count + kingfisher_count)
  (other_count / reserve.total) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_other_birds_percentage_l2176_217658


namespace NUMINAMATH_CALUDE_total_apples_l2176_217636

/-- Given 37 baskets with 17 apples each, prove that the total number of apples is 629. -/
theorem total_apples (num_baskets : ℕ) (apples_per_basket : ℕ) (h1 : num_baskets = 37) (h2 : apples_per_basket = 17) :
  num_baskets * apples_per_basket = 629 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l2176_217636


namespace NUMINAMATH_CALUDE_gallon_paint_cost_l2176_217664

def pints_needed : ℕ := 8
def pint_cost : ℚ := 8
def gallon_equivalent_pints : ℕ := 8
def savings : ℚ := 9

def total_pint_cost : ℚ := pints_needed * pint_cost

theorem gallon_paint_cost : 
  total_pint_cost - savings = 55 := by sorry

end NUMINAMATH_CALUDE_gallon_paint_cost_l2176_217664


namespace NUMINAMATH_CALUDE_diorama_time_factor_l2176_217681

def total_time : ℕ := 67
def building_time : ℕ := 49

theorem diorama_time_factor :
  ∃ (planning_time : ℕ) (factor : ℚ),
    planning_time + building_time = total_time ∧
    building_time = planning_time * factor - 5 ∧
    factor = 3 := by
  sorry

end NUMINAMATH_CALUDE_diorama_time_factor_l2176_217681


namespace NUMINAMATH_CALUDE_school_girls_count_l2176_217654

/-- Represents the number of girls in a school with the given conditions -/
def number_of_girls (total_students sample_size girl_boy_diff : ℕ) : ℕ :=
  (total_students * (sample_size / 2 - girl_boy_diff / 2)) / sample_size

/-- Theorem stating that under the given conditions, the number of girls in the school is 720 -/
theorem school_girls_count :
  number_of_girls 1600 200 20 = 720 := by
  sorry

end NUMINAMATH_CALUDE_school_girls_count_l2176_217654


namespace NUMINAMATH_CALUDE_slope_angle_sqrt3x_minus_y_plus3_l2176_217666

/-- The slope angle of the line √3x - y + 3 = 0 is π/3 -/
theorem slope_angle_sqrt3x_minus_y_plus3 :
  let line := {(x, y) : ℝ × ℝ | Real.sqrt 3 * x - y + 3 = 0}
  ∃ θ : ℝ, θ = π / 3 ∧ ∀ (x y : ℝ), (x, y) ∈ line → Real.tan θ = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_slope_angle_sqrt3x_minus_y_plus3_l2176_217666


namespace NUMINAMATH_CALUDE_vegetarians_count_l2176_217688

/-- Represents the eating habits of a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_non_veg : ℕ
  both : ℕ

/-- Calculates the total number of people who eat vegetarian food -/
def total_vegetarians (fd : FamilyDiet) : ℕ :=
  fd.only_veg + fd.both

/-- Theorem stating that the number of vegetarians in the given family is 20 -/
theorem vegetarians_count (fd : FamilyDiet) 
  (h1 : fd.only_veg = 11)
  (h2 : fd.only_non_veg = 6)
  (h3 : fd.both = 9) :
  total_vegetarians fd = 20 := by
  sorry

#eval total_vegetarians ⟨11, 6, 9⟩

end NUMINAMATH_CALUDE_vegetarians_count_l2176_217688


namespace NUMINAMATH_CALUDE_initial_toy_cost_l2176_217690

theorem initial_toy_cost (total_toys : ℕ) (total_cost : ℕ) (teddy_bears : ℕ) (teddy_cost : ℕ) (initial_toys : ℕ) :
  total_toys = initial_toys + teddy_bears →
  total_cost = teddy_bears * teddy_cost + initial_toys * 10 →
  teddy_bears = 20 →
  teddy_cost = 15 →
  initial_toys = 28 →
  total_cost = 580 →
  10 = total_cost / total_toys - (teddy_bears * teddy_cost) / initial_toys :=
by sorry

end NUMINAMATH_CALUDE_initial_toy_cost_l2176_217690


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2176_217665

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 / x + 1 / y = 1 / 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 4 / a + 1 / b = 1 / 2 → x + y ≤ a + b ∧ 
  ∃ c d : ℝ, c > 0 ∧ d > 0 ∧ 4 / c + 1 / d = 1 / 2 ∧ c + d = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2176_217665


namespace NUMINAMATH_CALUDE_parallel_vectors_m_l2176_217642

def vector_a : Fin 3 → ℝ := ![2, 4, 3]
def vector_b (m : ℝ) : Fin 3 → ℝ := ![4, 8, m]

def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, u i = k * v i

theorem parallel_vectors_m (m : ℝ) :
  parallel vector_a (vector_b m) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_l2176_217642


namespace NUMINAMATH_CALUDE_solve_age_problem_l2176_217659

def age_problem (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 9)

theorem solve_age_problem :
  ∃ (a b : ℕ), age_problem a b ∧ b = 39 :=
sorry

end NUMINAMATH_CALUDE_solve_age_problem_l2176_217659


namespace NUMINAMATH_CALUDE_date_book_cost_date_book_cost_value_l2176_217676

/-- Given the conditions of a real estate salesperson's promotional item purchase,
    prove that the cost of each date book is $0.375. -/
theorem date_book_cost (total_items : ℕ) (calendars : ℕ) (date_books : ℕ) 
                       (calendar_cost : ℚ) (total_spent : ℚ) : ℚ :=
  let date_book_cost := (total_spent - (calendar_cost * calendars)) / date_books
  by
    have h1 : total_items = calendars + date_books := by sorry
    have h2 : total_items = 500 := by sorry
    have h3 : calendars = 300 := by sorry
    have h4 : date_books = 200 := by sorry
    have h5 : calendar_cost = 3/4 := by sorry
    have h6 : total_spent = 300 := by sorry
    
    -- Prove that date_book_cost = 3/8
    sorry

#eval (300 : ℚ) - (3/4 * 300)
#eval ((300 : ℚ) - (3/4 * 300)) / 200

/-- The cost of each date book is $0.375. -/
theorem date_book_cost_value : 
  date_book_cost 500 300 200 (3/4) 300 = 3/8 := by sorry

end NUMINAMATH_CALUDE_date_book_cost_date_book_cost_value_l2176_217676


namespace NUMINAMATH_CALUDE_h_not_prime_l2176_217699

def h (n : ℕ+) : ℤ := n.val ^ 4 - 500 * n.val ^ 2 + 625

theorem h_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (h n)) := by
  sorry

end NUMINAMATH_CALUDE_h_not_prime_l2176_217699


namespace NUMINAMATH_CALUDE_trig_simplification_l2176_217621

theorem trig_simplification (x : ℝ) :
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) =
  (4 * (Real.cos x)^2 * Real.sin x) / (1 + 2 * Real.cos x * Real.cos (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2176_217621


namespace NUMINAMATH_CALUDE_fractional_inequality_solution_set_l2176_217660

theorem fractional_inequality_solution_set (x : ℝ) : 
  (x - 1) / (2 * x - 1) ≤ 0 ↔ 1/2 < x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_fractional_inequality_solution_set_l2176_217660


namespace NUMINAMATH_CALUDE_square_side_length_l2176_217646

theorem square_side_length (perimeter : ℝ) (h : perimeter = 100) : 
  ∃ (side_length : ℝ), side_length = 25 ∧ 4 * side_length = perimeter := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2176_217646


namespace NUMINAMATH_CALUDE_line_inclination_trig_identity_l2176_217691

/-- Given a line with equation x - 2y + 1 = 0 and inclination angle α, 
    prove that cos²α + sin(2α) = 8/5 -/
theorem line_inclination_trig_identity (α : ℝ) : 
  (∃ x y : ℝ, x - 2*y + 1 = 0 ∧ Real.tan α = 1/2) → 
  Real.cos α ^ 2 + Real.sin (2 * α) = 8/5 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_trig_identity_l2176_217691


namespace NUMINAMATH_CALUDE_rachels_homework_difference_l2176_217623

/-- Rachel's homework problem -/
theorem rachels_homework_difference (math_pages reading_pages biology_pages history_pages chemistry_pages : ℕ) : 
  math_pages = 15 → 
  reading_pages = 6 → 
  biology_pages = 126 → 
  history_pages = 22 → 
  chemistry_pages = 35 → 
  math_pages - reading_pages = 9 := by
sorry

end NUMINAMATH_CALUDE_rachels_homework_difference_l2176_217623
