import Mathlib

namespace range_a_theorem_l2407_240717

/-- The range of a satisfying both conditions -/
def range_a : Set ℝ :=
  {a | (0 ≤ a ∧ a < 1) ∨ (3 < a ∧ a < 4)}

/-- Condition 1: For all x ∈ ℝ, ax^2 + ax + 1 > 0 -/
def condition1 (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Condition 2: The standard equation of the hyperbola is (x^2)/(1-a) + (y^2)/(a-3) = 1 -/
def condition2 (a : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (1 - a) + y^2 / (a - 3) = 1

/-- The main theorem stating that if both conditions are satisfied, then a is in the specified range -/
theorem range_a_theorem (a : ℝ) :
  condition1 a ∧ condition2 a → a ∈ range_a :=
sorry

end range_a_theorem_l2407_240717


namespace tan_123_negative_l2407_240768

theorem tan_123_negative (a : ℝ) (h : Real.sin (123 * π / 180) = a) :
  Real.tan (123 * π / 180) < 0 := by
  sorry

end tan_123_negative_l2407_240768


namespace pentagon_square_side_ratio_l2407_240753

theorem pentagon_square_side_ratio :
  ∀ (p s : ℝ),
  p > 0 → s > 0 →
  5 * p = 20 →
  4 * s = 20 →
  p / s = 4 / 5 :=
by sorry

end pentagon_square_side_ratio_l2407_240753


namespace eighteenth_replacement_november_l2407_240793

/-- Represents months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Converts a number of months to a Month -/
def monthsToMonth (n : ℕ) : Month :=
  match n % 12 with
  | 0 => Month.December
  | 1 => Month.January
  | 2 => Month.February
  | 3 => Month.March
  | 4 => Month.April
  | 5 => Month.May
  | 6 => Month.June
  | 7 => Month.July
  | 8 => Month.August
  | 9 => Month.September
  | 10 => Month.October
  | _ => Month.November

/-- The month of the nth wheel replacement, given a 7-month cycle starting in January -/
def wheelReplacementMonth (n : ℕ) : Month :=
  monthsToMonth ((n - 1) * 7 + 1)

theorem eighteenth_replacement_november :
  wheelReplacementMonth 18 = Month.November := by
  sorry

end eighteenth_replacement_november_l2407_240793


namespace garden_breadth_l2407_240733

/-- The perimeter of a rectangle given its length and breadth -/
def perimeter (length breadth : ℝ) : ℝ := 2 * (length + breadth)

/-- Theorem: For a rectangular garden with perimeter 500 m and length 150 m, the breadth is 100 m -/
theorem garden_breadth :
  ∃ (breadth : ℝ), perimeter 150 breadth = 500 ∧ breadth = 100 := by
  sorry

end garden_breadth_l2407_240733


namespace alex_walk_distance_l2407_240742

def south_movement : ℝ := 50 + 15
def north_movement : ℝ := 30
def west_movement : ℝ := 80
def east_movement : ℝ := 40

def net_south : ℝ := south_movement - north_movement
def net_west : ℝ := west_movement - east_movement

theorem alex_walk_distance : 
  ∃ (length_AB : ℝ), length_AB = (net_south^2 + net_west^2).sqrt :=
sorry

end alex_walk_distance_l2407_240742


namespace sphere_volume_surface_ratio_l2407_240744

/-- The ratio of volume to surface area of a sphere with an inscribed regular hexagon -/
theorem sphere_volume_surface_ratio (area_hexagon : ℝ) (distance : ℝ) :
  area_hexagon = 3 * Real.sqrt 3 / 2 →
  distance = 2 * Real.sqrt 2 →
  ∃ (V S : ℝ), V / S = 1 :=
by sorry

end sphere_volume_surface_ratio_l2407_240744


namespace lcm_of_5_6_10_12_l2407_240701

theorem lcm_of_5_6_10_12 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 12)) = 60 := by sorry

end lcm_of_5_6_10_12_l2407_240701


namespace row_time_ratio_l2407_240736

/-- Proves the ratio of time taken to row up and down a river -/
theorem row_time_ratio (man_speed : ℝ) (stream_speed : ℝ)
  (h1 : man_speed = 24)
  (h2 : stream_speed = 12) :
  (man_speed - stream_speed) / (man_speed + stream_speed) = 1 / 3 := by
  sorry

#check row_time_ratio

end row_time_ratio_l2407_240736


namespace quadratic_roots_same_sign_a_range_l2407_240752

theorem quadratic_roots_same_sign_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2*x + 1 = 0 ∧ a * y^2 + 2*y + 1 = 0 ∧ (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0)) →
  0 < a ∧ a ≤ 1 :=
by sorry

end quadratic_roots_same_sign_a_range_l2407_240752


namespace gluten_free_pasta_cost_l2407_240745

theorem gluten_free_pasta_cost 
  (mustard_oil_quantity : ℕ)
  (mustard_oil_price : ℚ)
  (pasta_quantity : ℕ)
  (pasta_sauce_quantity : ℕ)
  (pasta_sauce_price : ℚ)
  (initial_money : ℚ)
  (remaining_money : ℚ)
  (h1 : mustard_oil_quantity = 2)
  (h2 : mustard_oil_price = 13)
  (h3 : pasta_quantity = 3)
  (h4 : pasta_sauce_quantity = 1)
  (h5 : pasta_sauce_price = 5)
  (h6 : initial_money = 50)
  (h7 : remaining_money = 7) :
  (initial_money - remaining_money - 
   (mustard_oil_quantity * mustard_oil_price + pasta_sauce_quantity * pasta_sauce_price)) / pasta_quantity = 4 := by
  sorry

end gluten_free_pasta_cost_l2407_240745


namespace inequality_preservation_l2407_240734

theorem inequality_preservation (a b : ℝ) : a < b → 1 - a > 1 - b := by sorry

end inequality_preservation_l2407_240734


namespace complex_number_problem_l2407_240771

theorem complex_number_problem (z : ℂ) (hz : z ≠ 0) :
  Complex.abs (z + 2) = 2 ∧ (z + 4 / z).im = 0 →
  z = -1 + Complex.I * Real.sqrt 3 ∨ z = -1 - Complex.I * Real.sqrt 3 := by
  sorry

end complex_number_problem_l2407_240771


namespace sum_squares_ge_sum_products_l2407_240755

theorem sum_squares_ge_sum_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end sum_squares_ge_sum_products_l2407_240755


namespace average_side_lengths_of_squares_l2407_240760

theorem average_side_lengths_of_squares (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 36) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 26 / 3 := by
  sorry

end average_side_lengths_of_squares_l2407_240760


namespace polygon_sides_l2407_240797

theorem polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → 
  (exterior_angle = 30) → 
  (n * exterior_angle = 360) → 
  n = 12 := by sorry

end polygon_sides_l2407_240797


namespace one_thirds_in_eleven_fifths_l2407_240770

theorem one_thirds_in_eleven_fifths : (11 / 5 : ℚ) / (1 / 3 : ℚ) = 33 / 5 := by sorry

end one_thirds_in_eleven_fifths_l2407_240770


namespace rational_sqrt_two_sum_l2407_240737

theorem rational_sqrt_two_sum (n : ℕ) : n ≥ 2 →
  (∃ a : ℝ, (∃ q : ℚ, a + Real.sqrt 2 = q) ∧ (∃ r : ℚ, a^n + Real.sqrt 2 = r)) ↔ n = 2 := by
  sorry

end rational_sqrt_two_sum_l2407_240737


namespace circle_center_l2407_240781

/-- The center of a circle given by the equation 4x^2 + 8x + 4y^2 - 12y + 20 = 0 is (-1, 3/2) -/
theorem circle_center (x y : ℝ) : 
  (4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 20 = 0) → 
  (∃ r : ℝ, (x + 1)^2 + (y - 3/2)^2 = r^2) := by
  sorry

end circle_center_l2407_240781


namespace min_c_value_l2407_240746

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_sum : a + b = c)
  (h_unique_solution : ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1501 ∧ ∃ (a' b' c' : ℕ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' < b' ∧ b' < c' ∧ a' + b' = c' ∧ c' = 1501 ∧
    ∃! (x y : ℝ), 3 * x + y = 3005 ∧ y = |x - a'| + |x - b'| + |x - c'| :=
by sorry

end min_c_value_l2407_240746


namespace smallest_five_digit_divisible_by_primes_l2407_240765

theorem smallest_five_digit_divisible_by_primes : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m → n ≤ m) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧
  n = 11550 :=
by sorry

#check smallest_five_digit_divisible_by_primes

end smallest_five_digit_divisible_by_primes_l2407_240765


namespace quadratic_equation_roots_l2407_240780

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a*x^2 + b*x + c = 0 ∧ a = 1 ∧ b = -5 ∧ c = 6 →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
by sorry

end quadratic_equation_roots_l2407_240780


namespace population_growth_rate_l2407_240709

theorem population_growth_rate (initial_population : ℝ) (final_population : ℝ) (second_year_decrease : ℝ) :
  initial_population = 20000 →
  final_population = 18750 →
  second_year_decrease = 0.25 →
  ∃ first_year_increase : ℝ,
    first_year_increase = 0.25 ∧
    final_population = initial_population * (1 + first_year_increase) * (1 - second_year_decrease) :=
by sorry

end population_growth_rate_l2407_240709


namespace min_value_T_l2407_240774

theorem min_value_T (a b c : ℝ) 
  (h1 : ∀ x : ℝ, 1/a * x^2 + 6*x + c ≥ 0)
  (h2 : a*b > 1)
  (h3 : ∃ x : ℝ, 1/a * x^2 + 6*x + c = 0) :
  1/(2*(a*b - 1)) + a*(b + 2*c)/(a*b - 1) ≥ 4 := by
  sorry

end min_value_T_l2407_240774


namespace complex_cube_equality_l2407_240787

theorem complex_cube_equality (a b c : ℝ) : 
  ((2 * a - b - c : ℂ) + (b - c) * Complex.I * Real.sqrt 3) ^ 3 = 
  ((2 * b - c - a : ℂ) + (c - a) * Complex.I * Real.sqrt 3) ^ 3 := by
  sorry

end complex_cube_equality_l2407_240787


namespace non_prime_sequence_300th_term_l2407_240776

/-- A sequence of positive integers with primes omitted -/
def non_prime_sequence : ℕ → ℕ := sorry

/-- The 300th term of the non-prime sequence -/
def term_300 : ℕ := 609

theorem non_prime_sequence_300th_term :
  non_prime_sequence 300 = term_300 := by sorry

end non_prime_sequence_300th_term_l2407_240776


namespace lcm_18_60_l2407_240705

theorem lcm_18_60 : Nat.lcm 18 60 = 180 := by
  sorry

end lcm_18_60_l2407_240705


namespace division_problem_l2407_240789

theorem division_problem (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) : 
  dividend = 62976 → quotient = 123 → divisor = 512 → 
  dividend = divisor * quotient ∧ dividend = 62976 := by
  sorry

end division_problem_l2407_240789


namespace jasper_chip_sales_l2407_240738

/-- Given the conditions of Jasper's sales, prove that he sold 27 bags of chips. -/
theorem jasper_chip_sales :
  ∀ (chips hotdogs drinks : ℕ),
    hotdogs = chips - 8 →
    drinks = hotdogs + 12 →
    drinks = 31 →
    chips = 27 :=
by
  sorry

end jasper_chip_sales_l2407_240738


namespace second_group_size_l2407_240751

theorem second_group_size (n : ℕ) : 
  (30 : ℝ) * 20 + n * 30 = (30 + n) * 24 → n = 20 := by sorry

end second_group_size_l2407_240751


namespace ratio_problem_l2407_240735

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : 
  y / x = 13 / 2 := by
  sorry

end ratio_problem_l2407_240735


namespace quadratic_function_properties_l2407_240703

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- Theorem statement
theorem quadratic_function_properties :
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ f x) ∧ -- Maximum value is 2
  (∃ (x : ℝ), f x = x + 1) ∧ -- Vertex lies on y = x + 1
  (f 3 = -2) ∧ -- Passes through (3, -2)
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 →
    f x ≤ 2 ∧ -- Maximum value in [0, 3] is 2
    f x ≥ -2 ∧ -- Minimum value in [0, 3] is -2
    (f x = 2 → x = 1) ∧ -- Maximum occurs at x = 1
    (f x = -2 → x = 3)) -- Minimum occurs at x = 3
  := by sorry

end quadratic_function_properties_l2407_240703


namespace provider_selection_ways_l2407_240739

def total_providers : ℕ := 25
def s_providers : ℕ := 6
def num_siblings : ℕ := 4

theorem provider_selection_ways : 
  (total_providers * s_providers * (total_providers - 2) * (total_providers - 3) = 75900) := by
  sorry

end provider_selection_ways_l2407_240739


namespace benjamin_weekly_miles_l2407_240762

/-- Calculates the total miles Benjamin walks in a week --/
def total_miles_walked : ℕ :=
  let work_distance := 6
  let dog_walk_distance := 2
  let friend_house_distance := 1
  let store_distance := 3
  let work_days := 5
  let dog_walks_per_day := 2
  let days_in_week := 7
  let store_visits := 2
  let friend_visits := 1

  let work_miles := work_distance * 2 * work_days
  let dog_walk_miles := dog_walk_distance * dog_walks_per_day * days_in_week
  let store_miles := store_distance * 2 * store_visits
  let friend_miles := friend_house_distance * 2 * friend_visits

  work_miles + dog_walk_miles + store_miles + friend_miles

theorem benjamin_weekly_miles :
  total_miles_walked = 95 := by
  sorry

end benjamin_weekly_miles_l2407_240762


namespace unique_solution_logarithmic_equation_l2407_240777

theorem unique_solution_logarithmic_equation (a b x : ℝ) :
  a > 0 ∧ b > 0 ∧ x > 1 ∧
  9 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 17 ∧
  (Real.log b / Real.log a) * (Real.log a / Real.log b) = 2 →
  a = Real.exp (Real.sqrt 2 * Real.log 10) ∧ b = 10 := by
sorry

end unique_solution_logarithmic_equation_l2407_240777


namespace words_lost_oz_l2407_240778

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 67

/-- The number of words lost when prohibiting one letter in a language with only one or two-letter words --/
def words_lost (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

theorem words_lost_oz :
  words_lost num_letters = 135 := by
  sorry

end words_lost_oz_l2407_240778


namespace m_values_l2407_240764

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem m_values (m : ℝ) : (A ∪ B m = A) ↔ m ∈ ({0, -1/2, 1/3} : Set ℝ) := by
  sorry

end m_values_l2407_240764


namespace mr_c_net_loss_l2407_240749

/-- Represents the value of a house and its transactions -/
structure HouseTransaction where
  initial_value : ℝ
  first_sale_loss_percent : ℝ
  second_sale_gain_percent : ℝ
  additional_tax : ℝ

/-- Calculates the net loss for Mr. C after two transactions -/
def net_loss (t : HouseTransaction) : ℝ :=
  let first_sale_price := t.initial_value * (1 - t.first_sale_loss_percent)
  let second_sale_price := first_sale_price * (1 + t.second_sale_gain_percent) + t.additional_tax
  second_sale_price - t.initial_value

/-- Theorem stating that Mr. C's net loss is $1560 -/
theorem mr_c_net_loss :
  let t : HouseTransaction := {
    initial_value := 8000,
    first_sale_loss_percent := 0.15,
    second_sale_gain_percent := 0.2,
    additional_tax := 200
  }
  net_loss t = 1560 := by
  sorry

end mr_c_net_loss_l2407_240749


namespace complex_cube_root_of_unity_sum_l2407_240706

theorem complex_cube_root_of_unity_sum (ω : ℂ) : 
  ω = (-1 + Complex.I * Real.sqrt 3) / 2 →
  ω^3 = 1 →
  ω^2 + ω + 1 = 0 →
  ((-1 + Complex.I * Real.sqrt 3) / 2)^4 + ((-1 - Complex.I * Real.sqrt 3) / 2)^4 = -1 :=
by
  sorry

end complex_cube_root_of_unity_sum_l2407_240706


namespace simple_interest_problem_l2407_240728

/-- Simple interest calculation -/
theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  interest = 4016.25 →
  rate = 11 →
  time = 5 →
  principal = interest / (rate * time / 100) →
  principal = 7302.27 := by
sorry

end simple_interest_problem_l2407_240728


namespace hamiltonian_circuit_theorem_l2407_240798

/-- Represents a rectangle on a grid with unit cells -/
structure GridRectangle where
  m : ℕ  -- width
  n : ℕ  -- height

/-- Determines if a Hamiltonian circuit exists on the grid rectangle -/
def has_hamiltonian_circuit (rect : GridRectangle) : Prop :=
  rect.m > 0 ∧ rect.n > 0 ∧ (Odd rect.m ∨ Odd rect.n)

/-- Calculates the length of the Hamiltonian circuit when it exists -/
def hamiltonian_circuit_length (rect : GridRectangle) : ℕ :=
  (rect.m + 1) * (rect.n + 1)

theorem hamiltonian_circuit_theorem (rect : GridRectangle) :
  has_hamiltonian_circuit rect ↔
    ∃ (path_length : ℕ), 
      path_length = hamiltonian_circuit_length rect ∧
      path_length > 0 :=
sorry

end hamiltonian_circuit_theorem_l2407_240798


namespace custom_baseball_caps_l2407_240782

theorem custom_baseball_caps (jack_circumference bill_circumference : ℝ)
  (h1 : jack_circumference = 12)
  (h2 : bill_circumference = 10)
  (h3 : ∃ f : ℝ, charlie_circumference = f * jack_circumference + 9)
  (h4 : bill_circumference = (2/3) * charlie_circumference) :
  ∃ f : ℝ, charlie_circumference = f * jack_circumference + 9 ∧ f = (1/2) :=
by
  sorry
where
  charlie_circumference : ℝ := bill_circumference / (2/3)

end custom_baseball_caps_l2407_240782


namespace height_weight_only_correlation_l2407_240741

-- Define the types of relationships
inductive Relationship
  | HeightWeight
  | DistanceTime
  | HeightVision
  | VolumeEdge

-- Define a property for correlation
def is_correlated (r : Relationship) : Prop :=
  match r with
  | Relationship.HeightWeight => True
  | _ => False

-- Define a property for functional relationships
def is_functional (r : Relationship) : Prop :=
  match r with
  | Relationship.DistanceTime => True
  | Relationship.VolumeEdge => True
  | _ => False

-- Theorem statement
theorem height_weight_only_correlation :
  ∀ r : Relationship, is_correlated r ↔ r = Relationship.HeightWeight ∧ ¬is_functional r :=
sorry

end height_weight_only_correlation_l2407_240741


namespace inverse_variation_problem_l2407_240786

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 1^3 = k) :
  a * 4^3 = k → a = 1/8 := by
  sorry

end inverse_variation_problem_l2407_240786


namespace curved_octagon_area_l2407_240748

/-- A closed curve composed of circular arcs centered on an octagon's vertices -/
structure CurvedOctagon where
  /-- Number of circular arcs -/
  n_arcs : ℕ
  /-- Length of each circular arc -/
  arc_length : ℝ
  /-- Side length of the regular octagon -/
  octagon_side : ℝ

/-- The area enclosed by the curved octagon -/
noncomputable def enclosed_area (co : CurvedOctagon) : ℝ :=
  sorry

/-- Theorem stating the enclosed area of a specific curved octagon -/
theorem curved_octagon_area :
  let co : CurvedOctagon := {
    n_arcs := 12,
    arc_length := 3 * Real.pi / 4,
    octagon_side := 3
  }
  enclosed_area co = 18 * (1 + Real.sqrt 2) + 81 * Real.pi / 8 :=
sorry

end curved_octagon_area_l2407_240748


namespace simplify_and_factorize_l2407_240769

theorem simplify_and_factorize (x : ℝ) : 
  3 * x^2 + 4 * x + 5 - (7 - 3 * x^2 - 5 * x) = (x + 2) * (6 * x - 1) := by
  sorry

end simplify_and_factorize_l2407_240769


namespace unique_solution_iff_m_not_neg_two_and_not_zero_l2407_240788

/-- Given an equation (m^2 + 2m + 3)x = 3(x + 2) + m - 4, it has a unique solution
    with respect to x if and only if m ≠ -2 and m ≠ 0 -/
theorem unique_solution_iff_m_not_neg_two_and_not_zero (m : ℝ) :
  (∃! x : ℝ, (m^2 + 2*m + 3)*x = 3*(x + 2) + m - 4) ↔ (m ≠ -2 ∧ m ≠ 0) :=
sorry

end unique_solution_iff_m_not_neg_two_and_not_zero_l2407_240788


namespace sqrt_expression_simplification_l2407_240759

theorem sqrt_expression_simplification :
  Real.sqrt 5 - Real.sqrt 20 + Real.sqrt 90 / Real.sqrt 2 = 2 * Real.sqrt 5 := by
  sorry

end sqrt_expression_simplification_l2407_240759


namespace quadratic_expression_value_l2407_240721

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 6) 
  (eq2 : 2 * x + 3 * y = 7) : 
  14 * x^2 + 25 * x * y + 14 * y^2 = 85 := by
  sorry

end quadratic_expression_value_l2407_240721


namespace greatest_two_digit_multiple_of_17_l2407_240729

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, n * 17 = 85 ∧ 
  (∀ m : ℕ, m * 17 ≤ 99 → m * 17 ≤ 85) := by
  sorry

end greatest_two_digit_multiple_of_17_l2407_240729


namespace sum_fifth_sixth_row20_l2407_240731

def pascal_triangle (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem sum_fifth_sixth_row20 : 
  pascal_triangle 20 4 + pascal_triangle 20 5 = 20349 := by sorry

end sum_fifth_sixth_row20_l2407_240731


namespace min_buses_required_l2407_240784

theorem min_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 325) (h2 : bus_capacity = 45) :
  ∃ (n : ℕ), n * bus_capacity ≥ total_students ∧ ∀ m : ℕ, m * bus_capacity ≥ total_students → m ≥ n ∧ n = 8 := by
  sorry

end min_buses_required_l2407_240784


namespace ferry_travel_time_difference_l2407_240740

/-- Represents the properties of a ferry --/
structure Ferry where
  baseSpeed : ℝ  -- Speed without current in km/h
  currentEffect : ℝ  -- Speed reduction due to current in km/h
  travelTime : ℝ  -- Travel time in hours
  routeLength : ℝ  -- Route length in km

/-- The problem setup --/
def ferryProblem : Prop := ∃ (p q : Ferry),
  -- Ferry p properties
  p.baseSpeed = 6 ∧
  p.currentEffect = 1 ∧
  p.travelTime = 3 ∧
  
  -- Ferry q properties
  q.baseSpeed = p.baseSpeed + 3 ∧
  q.currentEffect = p.currentEffect / 2 ∧
  q.routeLength = 2 * p.routeLength ∧
  
  -- Calculate effective speeds
  let pEffectiveSpeed := p.baseSpeed - p.currentEffect
  let qEffectiveSpeed := q.baseSpeed - q.currentEffect
  
  -- Calculate route lengths
  p.routeLength = pEffectiveSpeed * p.travelTime ∧
  
  -- Calculate q's travel time
  q.travelTime = q.routeLength / qEffectiveSpeed ∧
  
  -- The difference in travel time is approximately 0.5294 hours
  abs (q.travelTime - p.travelTime - 0.5294) < 0.0001

/-- The theorem to be proved --/
theorem ferry_travel_time_difference : ferryProblem := by
  sorry

end ferry_travel_time_difference_l2407_240740


namespace razorback_tshirt_profit_l2407_240750

/-- The Razorback T-shirt Shop problem -/
theorem razorback_tshirt_profit :
  let profit_per_shirt : ℕ := 9
  let shirts_sold : ℕ := 245
  let total_profit : ℕ := profit_per_shirt * shirts_sold
  total_profit = 2205 := by sorry

end razorback_tshirt_profit_l2407_240750


namespace julian_needs_more_legos_l2407_240710

/-- The number of legos Julian has -/
def julianLegos : ℕ := 400

/-- The number of legos required for one airplane model -/
def legosPerModel : ℕ := 240

/-- The number of airplane models Julian wants to make -/
def numModels : ℕ := 2

/-- The number of additional legos Julian needs -/
def additionalLegosNeeded : ℕ := 80

theorem julian_needs_more_legos : 
  julianLegos + additionalLegosNeeded = legosPerModel * numModels := by
  sorry

end julian_needs_more_legos_l2407_240710


namespace probability_y_l2407_240791

theorem probability_y (x y : Set Ω) (z : Set Ω → ℝ) 
  (hx : z x = 0.02)
  (hxy : z (x ∩ y) = 0.10)
  (hcond : z x / z y = 0.2) :
  z y = 0.5 := by
  sorry

end probability_y_l2407_240791


namespace estimated_y_at_25_l2407_240730

/-- Linear regression function -/
def linear_regression (x : ℝ) : ℝ := 0.5 * x - 0.81

/-- Theorem: The estimated value of y is 11.69 when x = 25 -/
theorem estimated_y_at_25 : linear_regression 25 = 11.69 := by
  sorry

end estimated_y_at_25_l2407_240730


namespace smallest_number_proof_l2407_240775

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_proof :
  ∃! x : ℕ, x > 0 ∧ 
    (∃ y : ℕ, y > 0 ∧ 
      x + y = 4728 ∧ 
      is_divisible_by (x + y) 27 ∧
      is_divisible_by (x + y) 35 ∧
      is_divisible_by (x + y) 25 ∧
      is_divisible_by (x + y) 21) ∧
    (∀ z : ℕ, z > 0 ∧ 
      (∃ w : ℕ, w > 0 ∧ 
        z + w = 4728 ∧ 
        is_divisible_by (z + w) 27 ∧
        is_divisible_by (z + w) 35 ∧
        is_divisible_by (z + w) 25 ∧
        is_divisible_by (z + w) 21) → 
      x ≤ z) ∧
  x = 4725 :=
sorry

end smallest_number_proof_l2407_240775


namespace elisa_lap_time_improvement_l2407_240757

/-- Calculates the improvement in lap time given current and previous swimming performance -/
def lap_time_improvement (current_laps : ℕ) (current_time : ℕ) (previous_laps : ℕ) (previous_time : ℕ) : ℚ :=
  (previous_time : ℚ) / (previous_laps : ℚ) - (current_time : ℚ) / (current_laps : ℚ)

/-- Proves that Elisa's lap time improvement is 0.5 minutes per lap -/
theorem elisa_lap_time_improvement :
  lap_time_improvement 15 30 20 50 = 1/2 := by sorry

end elisa_lap_time_improvement_l2407_240757


namespace subtraction_of_fractions_l2407_240785

theorem subtraction_of_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by sorry

end subtraction_of_fractions_l2407_240785


namespace gcd_lcm_240_360_l2407_240743

theorem gcd_lcm_240_360 : 
  (Nat.gcd 240 360 = 120) ∧ (Nat.lcm 240 360 = 720) := by
  sorry

end gcd_lcm_240_360_l2407_240743


namespace imaginary_part_of_product_l2407_240758

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_part_of_product : Complex.im ((1 - 2*i) * i) = 1 := by
  sorry

end imaginary_part_of_product_l2407_240758


namespace job_completion_time_l2407_240795

/-- The time taken by two workers to complete a job together, given their relative efficiencies and the time taken by one worker alone. -/
theorem job_completion_time 
  (efficiency_a : ℝ) 
  (efficiency_b : ℝ) 
  (time_a_alone : ℝ) 
  (h1 : efficiency_a = efficiency_b + 0.6 * efficiency_b) 
  (h2 : time_a_alone = 35) 
  : (1 / (1 / time_a_alone + efficiency_b / (efficiency_a * time_a_alone))) = 25 := by
  sorry

end job_completion_time_l2407_240795


namespace angle_A_value_max_area_l2407_240725

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C

/-- The theorem stating that A = π/3 given the condition -/
theorem angle_A_value (t : Triangle) (h : triangle_condition t) : t.A = π / 3 :=
sorry

/-- The theorem stating the maximum area when a = 4 -/
theorem max_area (t : Triangle) (h : triangle_condition t) (ha : t.a = 4) :
  (∀ t' : Triangle, triangle_condition t' → t'.a = 4 → 
    t'.b * t'.c * Real.sin t'.A / 2 ≤ 4 * Real.sqrt 3) ∧
  (∃ t' : Triangle, triangle_condition t' ∧ t'.a = 4 ∧ 
    t'.b * t'.c * Real.sin t'.A / 2 = 4 * Real.sqrt 3) :=
sorry

end angle_A_value_max_area_l2407_240725


namespace only_D_is_symmetric_l2407_240715

-- Define the type for shapes
inductive Shape
| A
| B
| C
| D
| E

-- Define a function to check if a shape is horizontally symmetric
def isHorizontallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.D => True
  | _ => False

-- Theorem statement
theorem only_D_is_symmetric :
  ∀ s : Shape, isHorizontallySymmetric s ↔ s = Shape.D :=
by
  sorry

end only_D_is_symmetric_l2407_240715


namespace total_cost_of_suits_l2407_240726

/-- The total cost of two suits, given the cost of an off-the-rack suit and the pricing rule for a tailored suit. -/
theorem total_cost_of_suits (off_the_rack_cost : ℕ) : 
  off_the_rack_cost = 300 →
  off_the_rack_cost + (3 * off_the_rack_cost + 200) = 1400 := by
sorry

end total_cost_of_suits_l2407_240726


namespace percent_profit_calculation_l2407_240714

/-- If the cost price of 60 articles is equal to the selling price of 40 articles,
    then the percent profit is 50%. -/
theorem percent_profit_calculation (C S : ℝ) 
  (h : C > 0) 
  (eq : 60 * C = 40 * S) : 
  (S - C) / C * 100 = 50 :=
by sorry

end percent_profit_calculation_l2407_240714


namespace parallel_vectors_x_value_l2407_240747

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, x)
  are_parallel a b → x = -4 := by
    sorry

end parallel_vectors_x_value_l2407_240747


namespace constant_expression_l2407_240702

theorem constant_expression (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hsum : x + y = 1) :
  x / (y^3 - 1) + y / (1 - x^3) + 2 * (x - y) / (x^2 * y^2 + 3) = 0 := by
  sorry

end constant_expression_l2407_240702


namespace businessmen_drinks_l2407_240761

theorem businessmen_drinks (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) :
  total = 30 →
  coffee = 15 →
  tea = 14 →
  both = 7 →
  total - (coffee + tea - both) = 8 := by
  sorry

end businessmen_drinks_l2407_240761


namespace f_intersects_axes_twice_l2407_240711

/-- The quadratic function f(x) = x^2 + 4x + 4 -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The number of intersection points between f(x) and the coordinate axes -/
def num_intersections : ℕ := 2

/-- Theorem stating that f(x) intersects the coordinate axes at exactly two points -/
theorem f_intersects_axes_twice :
  (∃! x : ℝ, f x = 0) ∧ (∃! y : ℝ, f 0 = y) ∧ num_intersections = 2 :=
sorry

end f_intersects_axes_twice_l2407_240711


namespace sequence_strictly_increasing_l2407_240763

theorem sequence_strictly_increasing (n : ℕ) (h : n ≥ 14) : 
  let a : ℕ → ℤ := λ k => k^4 - 20*k^2 - 10*k + 1
  a n > a (n-1) := by sorry

end sequence_strictly_increasing_l2407_240763


namespace vector_problem_l2407_240718

/-- Given vectors AB and BC in R², prove that -1/2 * AC equals the specified vector. -/
theorem vector_problem (AB BC : ℝ × ℝ) (h1 : AB = (3, 7)) (h2 : BC = (-2, 3)) :
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  (-1/2 : ℝ) • AC = (-1/2, -5) := by sorry

end vector_problem_l2407_240718


namespace sqrt_divided_by_two_is_ten_l2407_240794

theorem sqrt_divided_by_two_is_ten (x : ℝ) : (Real.sqrt x) / 2 = 10 → x = 400 := by
  sorry

end sqrt_divided_by_two_is_ten_l2407_240794


namespace complement_intersection_equals_set_l2407_240772

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {3, 4, 5}

-- Define set N
def N : Set Nat := {2, 3}

-- Theorem statement
theorem complement_intersection_equals_set :
  (U \ N) ∩ M = {4, 5} := by sorry

end complement_intersection_equals_set_l2407_240772


namespace cherry_pies_count_l2407_240719

/-- Given a total number of pies and a ratio of three types of pies,
    calculate the number of pies of the third type. -/
def calculate_cherry_pies (total_pies : ℕ) (ratio_apple : ℕ) (ratio_blueberry : ℕ) (ratio_cherry : ℕ) : ℕ :=
  let total_ratio := ratio_apple + ratio_blueberry + ratio_cherry
  let pies_per_part := total_pies / total_ratio
  ratio_cherry * pies_per_part

/-- Theorem stating that given 36 total pies and a ratio of 2:3:4 for apple, blueberry, and cherry pies,
    the number of cherry pies is 16. -/
theorem cherry_pies_count :
  calculate_cherry_pies 36 2 3 4 = 16 := by
  sorry

end cherry_pies_count_l2407_240719


namespace total_apples_l2407_240783

/-- Represents the number of apples Tessa has -/
def tessas_apples : ℕ := 4

/-- Represents the number of apples Anita gave to Tessa -/
def anitas_gift : ℕ := 5

/-- Theorem stating that Tessa's total apples is the sum of her initial apples and Anita's gift -/
theorem total_apples : tessas_apples + anitas_gift = 9 := by
  sorry

end total_apples_l2407_240783


namespace primality_extension_l2407_240716

theorem primality_extension (n : ℕ) (h1 : n ≥ 2) :
  (∀ k : ℕ, k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n)) :=
by sorry

end primality_extension_l2407_240716


namespace greatest_x_with_lcm_l2407_240756

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ m : ℕ, Nat.lcm x (Nat.lcm 15 21) = 105) → 
  x ≤ 105 ∧ 
  ∃ y : ℕ, y > 105 → ¬(∃ m : ℕ, Nat.lcm y (Nat.lcm 15 21) = 105) :=
by sorry

end greatest_x_with_lcm_l2407_240756


namespace set_a_constraint_l2407_240712

theorem set_a_constraint (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 2*x + a ≥ 0}
  1 ∉ A → a < 1 := by
  sorry

end set_a_constraint_l2407_240712


namespace unique_perfect_square_Q_l2407_240723

/-- The polynomial Q(x) = x^4 + 6x^3 + 13x^2 + 3x - 19 -/
def Q (x : ℤ) : ℤ := x^4 + 6*x^3 + 13*x^2 + 3*x - 19

/-- A function that checks if a given integer is a perfect square -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, m^2 = n

/-- Theorem stating that there is exactly one integer x for which Q(x) is a perfect square -/
theorem unique_perfect_square_Q : ∃! x : ℤ, is_perfect_square (Q x) :=
sorry

end unique_perfect_square_Q_l2407_240723


namespace sterling_total_questions_l2407_240766

/-- Represents the candy reward system and Sterling's performance --/
structure CandyReward where
  correct_reward : ℕ
  incorrect_penalty : ℕ
  correct_answers : ℕ
  total_questions : ℕ
  hypothetical_candy : ℕ

/-- Theorem stating that Sterling answered 9 questions in total --/
theorem sterling_total_questions 
  (reward : CandyReward) 
  (h1 : reward.correct_reward = 3)
  (h2 : reward.incorrect_penalty = 2)
  (h3 : reward.correct_answers = 7)
  (h4 : reward.hypothetical_candy = 31)
  (h5 : reward.hypothetical_candy = 
    (reward.correct_answers + 2) * reward.correct_reward - 
    (reward.total_questions - reward.correct_answers - 2) * reward.incorrect_penalty) : 
  reward.total_questions = 9 := by
  sorry

end sterling_total_questions_l2407_240766


namespace quadratic_has_two_distinct_real_roots_l2407_240722

theorem quadratic_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 - 5*x₁ - 1 = 0) ∧ (x₂^2 - 5*x₂ - 1 = 0) :=
sorry

end quadratic_has_two_distinct_real_roots_l2407_240722


namespace consecutive_digits_difference_l2407_240779

theorem consecutive_digits_difference (a : ℕ) (h : 1 ≤ a ∧ a ≤ 8) : 
  (100 * (a + 1) + 10 * a + (a - 1)) - (100 * (a - 1) + 10 * a + (a + 1)) = 198 := by
sorry

end consecutive_digits_difference_l2407_240779


namespace sticker_remainder_l2407_240724

theorem sticker_remainder (a b c : ℤ) 
  (ha : a % 5 = 1)
  (hb : b % 5 = 4)
  (hc : c % 5 = 3) : 
  (a + b + c) % 5 = 3 := by
  sorry

end sticker_remainder_l2407_240724


namespace jeremy_watermelons_l2407_240767

/-- The number of watermelons Jeremy eats per week -/
def jeremy_eats_per_week : ℕ := 3

/-- The number of watermelons Jeremy gives to his dad per week -/
def jeremy_gives_dad_per_week : ℕ := 2

/-- The number of weeks the watermelons will last -/
def weeks_watermelons_last : ℕ := 6

/-- The total number of watermelons Jeremy bought -/
def total_watermelons : ℕ := 30

theorem jeremy_watermelons :
  total_watermelons = (jeremy_eats_per_week + jeremy_gives_dad_per_week) * weeks_watermelons_last :=
by sorry

end jeremy_watermelons_l2407_240767


namespace quadratic_function_existence_l2407_240790

theorem quadratic_function_existence : ∃ (a b c : ℝ), 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) ∧
  |a * 2^2 + b * 2 + c| ≥ 7 := by
  sorry

end quadratic_function_existence_l2407_240790


namespace divisibility_and_ratio_theorem_l2407_240713

theorem divisibility_and_ratio_theorem (k : ℕ) (h : k > 1) :
  ∃ a b : ℕ, 1 < a ∧ a < b ∧ (a^2 + b^2 - 1) / (a * b) = k := by
  sorry

end divisibility_and_ratio_theorem_l2407_240713


namespace five_div_sqrt_five_times_one_over_sqrt_five_equals_one_l2407_240799

theorem five_div_sqrt_five_times_one_over_sqrt_five_equals_one :
  ∀ (sqrt_five : ℝ), sqrt_five > 0 → sqrt_five * sqrt_five = 5 →
  5 / sqrt_five * (1 / sqrt_five) = 1 := by
sorry

end five_div_sqrt_five_times_one_over_sqrt_five_equals_one_l2407_240799


namespace power_function_through_one_l2407_240732

theorem power_function_through_one (a : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^a
  f 1 = 1 := by sorry

end power_function_through_one_l2407_240732


namespace gwen_zoo_pictures_l2407_240704

/-- The number of pictures Gwen took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- The number of pictures Gwen took at the museum -/
def museum_pictures : ℕ := 29

/-- The number of pictures Gwen deleted -/
def deleted_pictures : ℕ := 15

/-- The number of pictures Gwen had after deleting -/
def remaining_pictures : ℕ := 55

/-- Theorem stating that the number of pictures Gwen took at the zoo is 41 -/
theorem gwen_zoo_pictures :
  zoo_pictures = 41 :=
by
  have h1 : zoo_pictures + museum_pictures - deleted_pictures = remaining_pictures :=
    sorry
  sorry

end gwen_zoo_pictures_l2407_240704


namespace app_total_cost_l2407_240707

/-- Calculates the total cost of an app with online access -/
def total_cost (initial_price : ℕ) (monthly_fee : ℕ) (months : ℕ) : ℕ :=
  initial_price + monthly_fee * months

/-- Proves that the total cost for the given conditions is $21 -/
theorem app_total_cost : total_cost 5 8 2 = 21 := by
  sorry

end app_total_cost_l2407_240707


namespace quadrilateral_area_is_2012021_5_l2407_240773

/-- The area of a quadrilateral with vertices at (1, 2), (1, 1), (4, 1), and (2009, 2010) -/
def quadrilateral_area : ℝ :=
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (4, 1)
  let D : ℝ × ℝ := (2009, 2010)
  -- Area calculation goes here
  0 -- Placeholder

/-- Theorem stating that the area of the quadrilateral is 2012021.5 square units -/
theorem quadrilateral_area_is_2012021_5 : quadrilateral_area = 2012021.5 := by
  sorry

end quadrilateral_area_is_2012021_5_l2407_240773


namespace stadium_length_l2407_240720

/-- Given a rectangular stadium with perimeter 800 meters and breadth 300 meters, its length is 100 meters. -/
theorem stadium_length (perimeter breadth : ℝ) (h1 : perimeter = 800) (h2 : breadth = 300) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter → perimeter / 2 - breadth = 100 := by
  sorry

end stadium_length_l2407_240720


namespace V₃_at_one_horner_equiv_f_l2407_240796

-- Define the polynomial f(x) = 3x^5 + 2x^3 - 8x + 5
def f (x : ℝ) : ℝ := 3 * x^5 + 2 * x^3 - 8 * x + 5

-- Define Horner's method for this polynomial
def horner (x : ℝ) : ℝ := (((((3 * x + 0) * x + 2) * x + 0) * x - 8) * x + 5)

-- Define V₃ in Horner's method
def V₃ (x : ℝ) : ℝ := ((3 * x + 0) * x + 2) * x + 0

-- Theorem: V₃(1) = 2
theorem V₃_at_one : V₃ 1 = 2 := by
  sorry

-- Prove that Horner's method is equivalent to the original polynomial
theorem horner_equiv_f : ∀ x, horner x = f x := by
  sorry

end V₃_at_one_horner_equiv_f_l2407_240796


namespace fraction_sum_equals_two_l2407_240700

theorem fraction_sum_equals_two : 
  (1 : ℚ) / 2 + (1 : ℚ) / 2 + (1 : ℚ) / 3 + (1 : ℚ) / 3 + (1 : ℚ) / 3 = 2 := by
  sorry

end fraction_sum_equals_two_l2407_240700


namespace three_card_draw_probability_l2407_240754

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of diamonds in a standard deck -/
def NumDiamonds : ℕ := 13

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Probability of drawing an Ace as the first card, a diamond as the second card, 
    and a Jack as the third card from a standard 52-card deck -/
theorem three_card_draw_probability : 
  (NumAces / StandardDeck) * (NumDiamonds / (StandardDeck - 1)) * (NumJacks / (StandardDeck - 2)) = 1 / 650 :=
by sorry

end three_card_draw_probability_l2407_240754


namespace alcohol_amount_l2407_240792

/-- Represents the amount of alcohol in liters -/
def alcohol : ℝ := 14

/-- Represents the amount of water in liters -/
def water : ℝ := 10.5

/-- The amount of water added to the mixture in liters -/
def water_added : ℝ := 7

/-- The initial ratio of alcohol to water -/
def initial_ratio : ℚ := 4/3

/-- The final ratio of alcohol to water after adding more water -/
def final_ratio : ℚ := 4/5

theorem alcohol_amount :
  (alcohol / water = initial_ratio) ∧
  (alcohol / (water + water_added) = final_ratio) →
  alcohol = 14 := by
sorry

end alcohol_amount_l2407_240792


namespace three_color_theorem_l2407_240708

theorem three_color_theorem (a b : ℕ) : 
  ∃ (f : ℤ → Fin 3), ∀ x : ℤ, f x ≠ f (x + a) ∧ f x ≠ f (x + b) := by
  sorry

end three_color_theorem_l2407_240708


namespace teal_sales_theorem_l2407_240727

/-- Represents the types of pies sold in the bakery -/
inductive PieType
| Pumpkin
| Custard

/-- Represents the properties of a pie -/
structure Pie where
  pieType : PieType
  slicesPerPie : Nat
  pricePerSlice : Nat
  piesCount : Nat

/-- Calculates the total sales for a given pie -/
def totalSales (pie : Pie) : Nat :=
  pie.slicesPerPie * pie.pricePerSlice * pie.piesCount

/-- Theorem: Teal's total sales from pumpkin and custard pies equal $340 -/
theorem teal_sales_theorem (pumpkinPie custardPie : Pie)
    (h_pumpkin : pumpkinPie.pieType = PieType.Pumpkin ∧
                 pumpkinPie.slicesPerPie = 8 ∧
                 pumpkinPie.pricePerSlice = 5 ∧
                 pumpkinPie.piesCount = 4)
    (h_custard : custardPie.pieType = PieType.Custard ∧
                 custardPie.slicesPerPie = 6 ∧
                 custardPie.pricePerSlice = 6 ∧
                 custardPie.piesCount = 5) :
    totalSales pumpkinPie + totalSales custardPie = 340 := by
  sorry

end teal_sales_theorem_l2407_240727
