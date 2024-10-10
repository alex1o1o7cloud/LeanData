import Mathlib

namespace intersection_nonempty_l1887_188709

def M (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 - 1 = k * (p.1 + 1)}

def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.2 = 0}

theorem intersection_nonempty (k : ℝ) : ∃ p : ℝ × ℝ, p ∈ M k ∩ N := by
  sorry

end intersection_nonempty_l1887_188709


namespace lcm_5_6_8_18_l1887_188741

theorem lcm_5_6_8_18 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 18)) = 360 := by
  sorry

end lcm_5_6_8_18_l1887_188741


namespace prob_different_suits_modified_deck_l1887_188776

/-- A deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- The probability of drawing two cards of different suits -/
def prob_different_suits (d : Deck) : ℚ :=
  (d.total_cards - d.cards_per_suit) / (d.total_cards - 1)

/-- The modified 40-card deck -/
def modified_deck : Deck :=
  { total_cards := 40
  , num_suits := 4
  , cards_per_suit := 10
  , h_total := rfl }

theorem prob_different_suits_modified_deck :
  prob_different_suits modified_deck = 10 / 13 := by
  sorry

end prob_different_suits_modified_deck_l1887_188776


namespace quadratic_inequality_l1887_188732

theorem quadratic_inequality (x : ℝ) : x ^ 2 - 4 * x - 21 ≤ 0 ↔ x ∈ Set.Icc (-3) 7 := by
  sorry

end quadratic_inequality_l1887_188732


namespace complex_modulus_sqrt_l1887_188739

theorem complex_modulus_sqrt (z : ℂ) (h : z^2 = -15 + 8*I) : Complex.abs z = Real.sqrt 17 := by
  sorry

end complex_modulus_sqrt_l1887_188739


namespace group_work_problem_l1887_188756

theorem group_work_problem (n : ℕ) (W_total : ℝ) : 
  (n : ℝ) * (W_total / 55) = ((n : ℝ) - 15) * (W_total / 60) → n = 165 := by
  sorry

end group_work_problem_l1887_188756


namespace trapezoid_height_l1887_188765

/-- The height of a trapezoid with specific properties -/
theorem trapezoid_height (a b : ℝ) (h_ab : 0 < a ∧ a < b) : 
  let height := a * b / (b - a)
  ∃ (x y : ℝ), 
    (x^2 + y^2 = a^2 + b^2) ∧ 
    ((b - a)^2 = x^2 + y^2 - x*y*Real.sqrt 2) ∧
    (x * y * Real.sqrt 2 = 2 * (b - a) * height) :=
by sorry

end trapezoid_height_l1887_188765


namespace dessert_preference_l1887_188746

theorem dessert_preference (total : Nat) (apple : Nat) (chocolate : Nat) (pumpkin : Nat) (none : Nat)
  (h1 : total = 50)
  (h2 : apple = 22)
  (h3 : chocolate = 17)
  (h4 : pumpkin = 10)
  (h5 : none = 15)
  (h6 : total ≥ apple + chocolate + pumpkin - none) :
  ∃ x : Nat, x = 7 ∧ x ≤ apple ∧ x ≤ chocolate ∧ x ≤ pumpkin ∧
  apple + chocolate + pumpkin - 2*x = total - none :=
by sorry

end dessert_preference_l1887_188746


namespace star_computation_l1887_188760

-- Define the * operation
def star (a b : ℚ) : ℚ := (a^2 - b^2) / (1 - a * b)

-- Theorem statement
theorem star_computation : star 1 (star 2 (star 3 4)) = -18 := by
  sorry

end star_computation_l1887_188760


namespace inequalities_propositions_l1887_188703

theorem inequalities_propositions :
  (∀ a b : ℝ, a * b > 0 → a > b → 1 / a < 1 / b) ∧
  (∀ a b : ℝ, a > abs b → a^2 > b^2) ∧
  (∃ a b c d : ℝ, a > b ∧ a > d ∧ a - c ≤ b - d) ∧
  (∃ a b m : ℝ, a < b ∧ m > 0 ∧ a / b ≥ (a + m) / (b + m)) :=
by sorry

end inequalities_propositions_l1887_188703


namespace rotation_volumes_equal_l1887_188708

/-- The volume obtained by rotating a region about the y-axis -/
noncomputable def rotationVolume (region : Set (ℝ × ℝ)) : ℝ := sorry

/-- The region enclosed by x^2 = 4y, x^2 = -4y, x = 4, and x = -4 -/
def region1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 = 4*p.2 ∨ p.1^2 = -4*p.2) ∧ (p.1 = 4 ∨ p.1 = -4)}

/-- The region defined by x^2 + y^2 ≤ 16, x^2 + (y-2)^2 ≥ 4, and x^2 + (y+2)^2 ≥ 4 -/
def region2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 16 ∧ p.1^2 + (p.2-2)^2 ≥ 4 ∧ p.1^2 + (p.2+2)^2 ≥ 4}

/-- The theorem stating that the volumes of rotation are equal -/
theorem rotation_volumes_equal : rotationVolume region1 = rotationVolume region2 := by
  sorry

end rotation_volumes_equal_l1887_188708


namespace equation_solution_l1887_188777

theorem equation_solution :
  ∃ x : ℚ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 63 / 14 := by
  sorry

end equation_solution_l1887_188777


namespace ellipse_equation_l1887_188721

theorem ellipse_equation (e : ℝ) (h_e : e = (2/5) * Real.sqrt 5) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    e = Real.sqrt (a^2 - b^2) / a ∧
    1^2 / b^2 + 0^2 / a^2 = 1 ∧
    (∀ x y : ℝ, x^2 / b^2 + y^2 / a^2 = 1 ↔ x^2 + (1/5) * y^2 = 1) :=
by sorry

end ellipse_equation_l1887_188721


namespace algebraic_simplification_l1887_188786

theorem algebraic_simplification (b : ℝ) : 3*b*(3*b^2 + 2*b - 1) - 2*b^2 = 9*b^3 + 4*b^2 - 3*b := by
  sorry

end algebraic_simplification_l1887_188786


namespace interest_rate_proof_l1887_188704

/-- Proves that the interest rate is 5% given the specified loan conditions -/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (interest : ℝ) :
  principal = 3000 →
  time = 5 →
  interest = principal - 2250 →
  (interest * 100) / (principal * time) = 5 := by
  sorry

end interest_rate_proof_l1887_188704


namespace expression_non_negative_lower_bound_achievable_l1887_188740

/-- The expression is always non-negative for real x and y -/
theorem expression_non_negative (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 := by
  sorry

/-- The lower bound of 0 is achievable -/
theorem lower_bound_achievable :
  ∃ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 = 0 := by
  sorry

end expression_non_negative_lower_bound_achievable_l1887_188740


namespace complement_of_union_l1887_188723

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_of_union :
  (M ∪ N)ᶜ = {1, 6} :=
by sorry

end complement_of_union_l1887_188723


namespace total_legs_equals_1564_l1887_188766

/-- Calculates the total number of legs of all animals owned by Mark -/
def totalLegs (numKangaroos : ℕ) : ℕ :=
  let numGoats := 3 * numKangaroos
  let numSpiders := 2 * numGoats
  let numBirds := numSpiders / 2
  let kangarooLegs := 2 * numKangaroos
  let goatLegs := 4 * numGoats
  let spiderLegs := 8 * numSpiders
  let birdLegs := 2 * numBirds
  kangarooLegs + goatLegs + spiderLegs + birdLegs

/-- Theorem stating that the total number of legs of all Mark's animals is 1564 -/
theorem total_legs_equals_1564 : totalLegs 23 = 1564 := by
  sorry

end total_legs_equals_1564_l1887_188766


namespace jacks_burgers_l1887_188755

/-- Given Jack's barbecue sauce recipe and usage, prove how many burgers he can make. -/
theorem jacks_burgers :
  -- Total sauce
  let total_sauce : ℚ := 3 + 1 + 1

  -- Sauce per burger
  let sauce_per_burger : ℚ := 1 / 4

  -- Sauce per pulled pork sandwich
  let sauce_per_pps : ℚ := 1 / 6

  -- Number of pulled pork sandwiches
  let num_pps : ℕ := 18

  -- Sauce used for pulled pork sandwiches
  let sauce_for_pps : ℚ := sauce_per_pps * num_pps

  -- Remaining sauce for burgers
  let remaining_sauce : ℚ := total_sauce - sauce_for_pps

  -- Number of burgers Jack can make
  ↑(remaining_sauce / sauce_per_burger).floor = 8 :=
by
  sorry

end jacks_burgers_l1887_188755


namespace factorial_inequality_l1887_188758

theorem factorial_inequality (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n ≤ m) :
  (2^n : ℝ) * (n.factorial : ℝ) ≤ ((m+n).factorial : ℝ) / ((m-n).factorial : ℝ) ∧
  ((m+n).factorial : ℝ) / ((m-n).factorial : ℝ) ≤ ((m^2 + m : ℝ)^n : ℝ) := by
  sorry

end factorial_inequality_l1887_188758


namespace system_solution_l1887_188724

theorem system_solution (x y : ℝ) : 
  (x^x = y ∧ x^y = y^x) ↔ ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 2 ∧ y = 4)) :=
sorry

end system_solution_l1887_188724


namespace maximize_product_l1887_188797

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 100) :
  x^4 * y^6 ≤ 40^4 * 60^6 ∧ 
  (x^4 * y^6 = 40^4 * 60^6 ↔ x = 40 ∧ y = 60) :=
sorry

end maximize_product_l1887_188797


namespace neighborhood_cable_cost_l1887_188712

/-- Calculates the total cost of cable for a neighborhood given the street layout and cable requirements. -/
theorem neighborhood_cable_cost
  (east_west_streets : ℕ)
  (east_west_length : ℝ)
  (north_south_streets : ℕ)
  (north_south_length : ℝ)
  (cable_per_street_mile : ℝ)
  (cable_cost_per_mile : ℝ)
  (h1 : east_west_streets = 18)
  (h2 : east_west_length = 2)
  (h3 : north_south_streets = 10)
  (h4 : north_south_length = 4)
  (h5 : cable_per_street_mile = 5)
  (h6 : cable_cost_per_mile = 2000) :
  (east_west_streets * east_west_length + north_south_streets * north_south_length) *
  cable_per_street_mile * cable_cost_per_mile = 760000 := by
  sorry


end neighborhood_cable_cost_l1887_188712


namespace equation_solution_set_l1887_188789

theorem equation_solution_set : 
  {x : ℝ | |x^2 - 5*x + 6| = x + 2} = {3 + Real.sqrt 5, 3 - Real.sqrt 5} := by
sorry

end equation_solution_set_l1887_188789


namespace select_team_count_l1887_188747

/-- The number of players in the basketball team -/
def total_players : ℕ := 12

/-- The number of players to be selected for the team -/
def team_size : ℕ := 5

/-- The number of twins in the team -/
def num_twins : ℕ := 2

/-- Calculates the binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of ways to select the team with the given conditions -/
def select_team : ℕ := binomial total_players team_size - binomial (total_players - num_twins) (team_size - num_twins)

theorem select_team_count : select_team = 672 := by
  sorry

end select_team_count_l1887_188747


namespace negative_fraction_identification_l1887_188775

-- Define a predicate for negative fractions
def is_negative_fraction (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ) ∧ x < 0

-- Theorem statement
theorem negative_fraction_identification :
  is_negative_fraction (-0.7) ∧
  ¬is_negative_fraction (1/2) ∧
  ¬is_negative_fraction (-π) ∧
  ¬is_negative_fraction (-3/3) :=
by sorry

end negative_fraction_identification_l1887_188775


namespace percentage_calculation_l1887_188731

theorem percentage_calculation (x : ℝ) (h : x ≠ 0) : (x + 0.5 * x) / (0.75 * x) = 2 := by
  sorry

end percentage_calculation_l1887_188731


namespace cone_volume_from_cylinder_l1887_188794

theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cylinder_volume := π * r^2 * h
  let cone_volume := (1/3) * π * r^2 * h
  cylinder_volume = 108 * π → cone_volume = 36 * π := by
  sorry

end cone_volume_from_cylinder_l1887_188794


namespace more_heads_probability_l1887_188701

def coin_prob : ℚ := 2/3

def num_flips : ℕ := 5

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

def more_heads_prob : ℚ :=
  binomial_probability num_flips 3 coin_prob +
  binomial_probability num_flips 4 coin_prob +
  binomial_probability num_flips 5 coin_prob

theorem more_heads_probability :
  more_heads_prob = 64/81 := by sorry

end more_heads_probability_l1887_188701


namespace dividend_calculation_l1887_188757

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 19)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end dividend_calculation_l1887_188757


namespace base_9_minus_b_multiple_of_7_l1887_188782

def base_9_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

def is_multiple_of (a b : Int) : Prop :=
  ∃ k : Int, a = b * k

theorem base_9_minus_b_multiple_of_7 (b : Int) :
  (0 ≤ b) →
  (b ≤ 9) →
  (is_multiple_of (base_9_to_decimal [2, 7, 6, 4, 5, 1, 3] - b) 7) →
  b = 0 := by
  sorry

end base_9_minus_b_multiple_of_7_l1887_188782


namespace collinear_points_x_value_l1887_188751

/-- Given three points in a 2D plane, checks if they are collinear --/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem: If A(-1, -2), B(4, 8), and C(5, x) are collinear, then x = 10 --/
theorem collinear_points_x_value :
  collinear (-1) (-2) 4 8 5 x → x = 10 :=
by
  sorry

#check collinear_points_x_value

end collinear_points_x_value_l1887_188751


namespace find_m_l1887_188725

def U : Set Nat := {1, 2, 3, 4}

def A (m : ℤ) : Set Nat := {x ∈ U | x^2 - 5*x + m = 0}

theorem find_m : ∃ m : ℤ, (U \ A m) = {1, 4} ∧ m = 6 := by sorry

end find_m_l1887_188725


namespace tan_alpha_value_l1887_188754

theorem tan_alpha_value (α : ℝ) (h : Real.sin α + 2 * Real.cos α = Real.sqrt 10 / 2) : 
  Real.tan α = -1/3 := by sorry

end tan_alpha_value_l1887_188754


namespace systematic_sampling_interval_l1887_188745

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The sampling interval for a population of 800 and sample size of 40 is 20 -/
theorem systematic_sampling_interval :
  samplingInterval 800 40 = 20 := by
  sorry

end systematic_sampling_interval_l1887_188745


namespace base_2_representation_of_123_l1887_188707

theorem base_2_representation_of_123 : 
  ∃ (a b c d e f g : ℕ), 
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1) ∧
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end base_2_representation_of_123_l1887_188707


namespace sum_smallest_largest_prime_l1887_188717

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def primes_in_range (a b : ℕ) : Set ℕ :=
  {n : ℕ | a ≤ n ∧ n ≤ b ∧ is_prime n}

theorem sum_smallest_largest_prime :
  let P := primes_in_range 1 50
  ∃ (p q : ℕ), p ∈ P ∧ q ∈ P ∧
    (∀ x ∈ P, p ≤ x) ∧
    (∀ x ∈ P, x ≤ q) ∧
    p + q = 49 :=
sorry

end sum_smallest_largest_prime_l1887_188717


namespace black_water_bottles_l1887_188773

theorem black_water_bottles (red : ℕ) (blue : ℕ) (taken_out : ℕ) (left : ℕ) :
  red = 2 →
  blue = 4 →
  taken_out = 5 →
  left = 4 →
  ∃ black : ℕ, red + black + blue = taken_out + left ∧ black = 3 :=
by sorry

end black_water_bottles_l1887_188773


namespace dog_treats_duration_l1887_188710

theorem dog_treats_duration (treats_per_day : ℕ) (cost_per_treat : ℚ) (total_spent : ℚ) : 
  treats_per_day = 2 → cost_per_treat = 1/10 → total_spent = 6 → 
  (total_spent / cost_per_treat) / treats_per_day = 30 := by
  sorry

end dog_treats_duration_l1887_188710


namespace shoe_multiple_l1887_188770

theorem shoe_multiple (bonny_shoes becky_shoes bobby_shoes : ℕ) : 
  bonny_shoes = 13 →
  bonny_shoes = 2 * becky_shoes - 5 →
  bobby_shoes = 27 →
  ∃ m : ℕ, m * becky_shoes = bobby_shoes ∧ m = 3 := by
sorry

end shoe_multiple_l1887_188770


namespace trishas_walk_distance_l1887_188787

theorem trishas_walk_distance :
  let distance_hotel_to_postcard : ℚ := 0.1111111111111111
  let distance_postcard_to_tshirt : ℚ := 0.1111111111111111
  let distance_tshirt_to_hotel : ℚ := 0.6666666666666666
  distance_hotel_to_postcard + distance_postcard_to_tshirt + distance_tshirt_to_hotel = 0.8888888888888888 := by
  sorry

end trishas_walk_distance_l1887_188787


namespace max_plus_min_equals_two_l1887_188727

noncomputable def f (x : ℝ) : ℝ := (2^x + 1)^2 / (2^x * x) + 1

def interval := {x : ℝ | (x ∈ Set.Icc (-2018) 0 ∧ x ≠ 0) ∨ (x ∈ Set.Ioc 0 2018)}

theorem max_plus_min_equals_two :
  ∃ (M N : ℝ), (∀ x ∈ interval, f x ≤ M) ∧
               (∀ x ∈ interval, N ≤ f x) ∧
               (∃ x₁ ∈ interval, f x₁ = M) ∧
               (∃ x₂ ∈ interval, f x₂ = N) ∧
               M + N = 2 := by
  sorry

end max_plus_min_equals_two_l1887_188727


namespace seating_problem_l1887_188738

/-- The number of ways to seat people on a bench with given constraints -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) (min_gap : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of seating arrangements for the given problem -/
theorem seating_problem : seating_arrangements 9 3 2 = 60 := by
  sorry

end seating_problem_l1887_188738


namespace cube_sum_given_sum_and_product_l1887_188716

theorem cube_sum_given_sum_and_product (x y : ℝ) :
  x + y = 10 → x * y = 15 → x^3 + y^3 = 550 := by sorry

end cube_sum_given_sum_and_product_l1887_188716


namespace A_n_squared_value_l1887_188792

theorem A_n_squared_value (n : ℕ) : (n.choose 2 = 15) → (n * (n - 1) = 30) := by
  sorry

end A_n_squared_value_l1887_188792


namespace quadratic_is_square_of_binomial_l1887_188711

/-- If ax^2 + 28x + 9 is the square of a binomial, then a = 196/9 -/
theorem quadratic_is_square_of_binomial (a : ℚ) : 
  (∃ p q : ℚ, ∀ x : ℚ, a * x^2 + 28 * x + 9 = (p * x + q)^2) → 
  a = 196 / 9 := by
sorry

end quadratic_is_square_of_binomial_l1887_188711


namespace function_periodicity_l1887_188700

/-- A function satisfying the given functional equation is periodic with period 8. -/
theorem function_periodicity (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) + f (x - 1) = Real.sqrt 2 * f x) : 
  ∀ x : ℝ, f (x + 8) = f x := by
  sorry

end function_periodicity_l1887_188700


namespace binary_67_l1887_188795

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

/-- Theorem: The binary representation of 67 is [1,0,0,0,0,1,1] -/
theorem binary_67 : toBinary 67 = [1,0,0,0,0,1,1] := by
  sorry

end binary_67_l1887_188795


namespace other_number_is_64_l1887_188791

/-- Given two positive integers with specific LCM and HCF, prove that one is 64 -/
theorem other_number_is_64 (A B : ℕ+) (h1 : A = 48) 
  (h2 : Nat.lcm A B = 192) (h3 : Nat.gcd A B = 16) : B = 64 := by
  sorry

end other_number_is_64_l1887_188791


namespace factorization_equality_l1887_188749

theorem factorization_equality (x : ℝ) :
  (x^2 + 3*x - 3) * (x^2 + 3*x + 1) - 5 = (x + 1) * (x + 2) * (x + 4) * (x - 1) := by
  sorry

end factorization_equality_l1887_188749


namespace quadratic_no_real_roots_l1887_188774

/-- A quadratic function f(x) = x^2 + 2x + a has no real roots if and only if a > 1 -/
theorem quadratic_no_real_roots (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a ≠ 0) ↔ a > 1 := by
  sorry

end quadratic_no_real_roots_l1887_188774


namespace sum_of_digits_for_four_elevenths_l1887_188736

theorem sum_of_digits_for_four_elevenths : ∃ (x y : ℕ), 
  (x < 10 ∧ y < 10) ∧ 
  (4 : ℚ) / 11 = (x * 10 + y : ℚ) / 99 ∧
  x + y = 9 := by
sorry

end sum_of_digits_for_four_elevenths_l1887_188736


namespace inequality_solution_l1887_188763

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x) ↔ x = 1 := by
  sorry

end inequality_solution_l1887_188763


namespace train_speed_l1887_188743

/-- Given a train that travels 80 km in 40 minutes, prove its speed is 120 kmph -/
theorem train_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) : 
  distance = 80 ∧ time_minutes = 40 → speed = distance / (time_minutes / 60) → speed = 120 := by
  sorry

end train_speed_l1887_188743


namespace carmen_pets_difference_l1887_188719

def initial_cats : ℕ := 28
def initial_dogs : ℕ := 18
def cats_given_up : ℕ := 3

theorem carmen_pets_difference :
  initial_cats - cats_given_up - initial_dogs = 7 :=
by
  sorry

end carmen_pets_difference_l1887_188719


namespace smallest_integer_l1887_188713

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 84) :
  ∀ c : ℕ, c > 0 ∧ Nat.lcm a c / Nat.gcd a c = 84 → b ≤ c → b = 35 :=
sorry

end smallest_integer_l1887_188713


namespace geometric_sequence_first_term_l1887_188742

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_second_term : a 1 = 3)
  (h_fourth_term : a 3 = 12) :
  a 0 = 2 * Real.sqrt 3 :=
sorry

end geometric_sequence_first_term_l1887_188742


namespace constant_c_value_l1887_188771

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end constant_c_value_l1887_188771


namespace parallel_line_through_point_l1887_188729

/-- Two lines are parallel if they do not intersect -/
def Parallel (l1 l2 : Set Point) : Prop := l1 ∩ l2 = ∅

/-- A point lies on a line if it is a member of the line's point set -/
def PointOnLine (p : Point) (l : Set Point) : Prop := p ∈ l

theorem parallel_line_through_point 
  (l l₁ : Set Point) (M : Point) 
  (h_parallel : Parallel l l₁)
  (h_M_not_on_l : ¬ PointOnLine M l)
  (h_M_not_on_l₁ : ¬ PointOnLine M l₁) :
  ∃ l₂ : Set Point, Parallel l₂ l ∧ Parallel l₂ l₁ ∧ PointOnLine M l₂ :=
sorry

end parallel_line_through_point_l1887_188729


namespace min_sum_4x4x4_dice_cube_l1887_188796

/-- Represents a 4x4x4 cube made of dice -/
structure LargeCube where
  size : Nat
  total_dice : Nat
  opposite_face_sum : Nat

/-- Calculates the minimum visible sum on the large cube -/
def min_visible_sum (c : LargeCube) : Nat :=
  sorry

/-- Theorem stating the minimum visible sum for a 4x4x4 cube of dice -/
theorem min_sum_4x4x4_dice_cube :
  ∀ c : LargeCube, 
    c.size = 4 → 
    c.total_dice = 64 → 
    c.opposite_face_sum = 7 → 
    min_visible_sum c = 144 :=
by sorry

end min_sum_4x4x4_dice_cube_l1887_188796


namespace calculation_proof_l1887_188753

theorem calculation_proof : (3.64 - 2.1) * 1.5 = 2.31 := by
  sorry

end calculation_proof_l1887_188753


namespace test_scores_l1887_188772

/-- Represents the score of a test taker -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  total_questions : Nat
  h_sum : correct + unanswered + incorrect = total_questions

/-- Calculates the score based on the test results -/
def calculate_score (ts : TestScore) : Nat :=
  4 * ts.correct + 2 * ts.unanswered

/-- Checks if a score is possible given the test parameters -/
def is_possible_score (score : Nat) : Prop :=
  ∃ (ts : TestScore), ts.total_questions = 30 ∧ calculate_score ts = score

theorem test_scores :
  is_possible_score 116 ∧
  ¬is_possible_score 117 ∧
  is_possible_score 118 ∧
  ¬is_possible_score 119 ∧
  is_possible_score 120 :=
sorry

end test_scores_l1887_188772


namespace antonette_overall_score_l1887_188750

/-- Calculates the overall score percentage on a combined test, given individual test scores and problem counts. -/
def overall_score (score1 score2 score3 : ℚ) (problems1 problems2 problems3 : ℕ) : ℚ :=
  (score1 * problems1 + score2 * problems2 + score3 * problems3) / (problems1 + problems2 + problems3)

/-- Rounds a rational number to the nearest integer. -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem antonette_overall_score :
  let score1 : ℚ := 70/100
  let score2 : ℚ := 80/100
  let score3 : ℚ := 90/100
  let problems1 : ℕ := 10
  let problems2 : ℕ := 20
  let problems3 : ℕ := 30
  round_to_nearest (overall_score score1 score2 score3 problems1 problems2 problems3 * 100) = 83 := by
  sorry

end antonette_overall_score_l1887_188750


namespace f_composition_at_pi_l1887_188790

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 1 else Real.sin x - 2

theorem f_composition_at_pi : f (f Real.pi) = -3/4 := by
  sorry

end f_composition_at_pi_l1887_188790


namespace line_perpendicular_to_plane_l1887_188744

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State the theorem
theorem line_perpendicular_to_plane 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_parallel : parallel m n)
  (h_perpendicular : perpendicular m β) :
  perpendicular n β := by
  sorry

end line_perpendicular_to_plane_l1887_188744


namespace three_integers_sum_and_reciprocals_l1887_188764

theorem three_integers_sum_and_reciprocals (a b c : ℕ+) : 
  (a + b + c : ℕ) = 15 ∧ 
  (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 71 / 105) → 
  ({a, b, c} : Finset ℕ+) = {3, 5, 7} := by
sorry

end three_integers_sum_and_reciprocals_l1887_188764


namespace cody_initial_tickets_cody_initial_tickets_proof_l1887_188778

/-- Theorem: Cody's initial number of tickets
Given:
- Cody lost 6.0 tickets
- Cody spent 25.0 tickets
- Cody has 18 tickets left
Prove: Cody's initial number of tickets was 49.0
-/
theorem cody_initial_tickets : ℝ → Prop :=
  fun initial_tickets =>
    let lost_tickets : ℝ := 6.0
    let spent_tickets : ℝ := 25.0
    let remaining_tickets : ℝ := 18.0
    initial_tickets = lost_tickets + spent_tickets + remaining_tickets
    ∧ initial_tickets = 49.0

/-- Proof of the theorem -/
theorem cody_initial_tickets_proof : cody_initial_tickets 49.0 := by
  sorry

end cody_initial_tickets_cody_initial_tickets_proof_l1887_188778


namespace caramel_distribution_solution_l1887_188733

def caramel_distribution (a b c d : ℕ) : Prop :=
  a + b + c + d = 26 ∧
  ∃ (x y : ℕ),
    a = x + y ∧
    b = 2 * x ∧
    c = x + y ∧
    d = x + (2 * y + x) ∧
    x > 0 ∧ y > 0

theorem caramel_distribution_solution :
  caramel_distribution 5 6 5 10 :=
sorry

end caramel_distribution_solution_l1887_188733


namespace sister_brother_product_is_twelve_l1887_188752

/-- Represents a family with siblings -/
structure Family :=
  (num_sisters : ℕ)
  (num_brothers : ℕ)

/-- Calculates the product of sisters and brothers for a sister in the family -/
def sister_brother_product (f : Family) : ℕ :=
  (f.num_sisters - 1) * f.num_brothers

/-- Theorem stating that for a family where one sibling has 4 sisters and 4 brothers,
    the product of sisters and brothers for any sister is 12 -/
theorem sister_brother_product_is_twelve (f : Family) 
  (h : f.num_sisters = 4 ∧ f.num_brothers = 4) : 
  sister_brother_product f = 12 := by
  sorry

#eval sister_brother_product ⟨4, 4⟩

end sister_brother_product_is_twelve_l1887_188752


namespace bowl_capacity_l1887_188726

/-- Given a bowl filled with oil and vinegar, prove its capacity. -/
theorem bowl_capacity (oil_density vinegar_density : ℝ)
                      (oil_fraction vinegar_fraction : ℝ)
                      (total_weight : ℝ) :
  oil_density = 5 →
  vinegar_density = 4 →
  oil_fraction = 2/3 →
  vinegar_fraction = 1/3 →
  total_weight = 700 →
  oil_fraction * oil_density + vinegar_fraction * vinegar_density = total_weight / 150 :=
by sorry

end bowl_capacity_l1887_188726


namespace fraction_inequality_l1887_188761

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end fraction_inequality_l1887_188761


namespace absolute_value_fraction_sum_not_one_l1887_188793

theorem absolute_value_fraction_sum_not_one (a b : ℝ) (h : a * b ≠ 0) :
  |a| / a + |b| / b ≠ 1 := by
  sorry

end absolute_value_fraction_sum_not_one_l1887_188793


namespace equation_solution_l1887_188785

theorem equation_solution (x : Real) :
  x ∈ Set.Ioo (-π / 2) 0 →
  (Real.sqrt 3 / Real.sin x) + (1 / Real.cos x) = 4 →
  x = -4 * π / 9 := by
  sorry

end equation_solution_l1887_188785


namespace some_athletes_not_honor_society_l1887_188734

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Athlete : U → Prop)
variable (Disciplined : U → Prop)
variable (HonorSocietyMember : U → Prop)

-- Define the given conditions
variable (some_athletes_not_disciplined : ∃ x, Athlete x ∧ ¬Disciplined x)
variable (all_honor_society_disciplined : ∀ x, HonorSocietyMember x → Disciplined x)

-- State the theorem
theorem some_athletes_not_honor_society :
  ∃ x, Athlete x ∧ ¬HonorSocietyMember x :=
sorry

end some_athletes_not_honor_society_l1887_188734


namespace line_does_not_intersect_curve_l1887_188706

/-- The function representing the curve y = (|x|-1)/(|x-1|) -/
noncomputable def f (x : ℝ) : ℝ := (abs x - 1) / (abs (x - 1))

/-- The theorem stating the condition for non-intersection -/
theorem line_does_not_intersect_curve (m : ℝ) :
  (∀ x : ℝ, m * x ≠ f x) ↔ (-1 ≤ m ∧ m < -3 + 2 * Real.sqrt 2) :=
sorry

end line_does_not_intersect_curve_l1887_188706


namespace angle_slope_relationship_l1887_188735

theorem angle_slope_relationship (α k : ℝ) :
  (k = Real.tan α) →
  (α < π / 3 → k < Real.sqrt 3) ∧
  ¬(k < Real.sqrt 3 → α < π / 3) :=
sorry

end angle_slope_relationship_l1887_188735


namespace complex_point_in_fourth_quadrant_l1887_188730

theorem complex_point_in_fourth_quadrant (a b : ℝ) :
  (a^2 - 4*a + 5 > 0) ∧ (-b^2 + 2*b - 6 < 0) :=
by
  sorry

#check complex_point_in_fourth_quadrant

end complex_point_in_fourth_quadrant_l1887_188730


namespace proposition_and_variants_l1887_188748

theorem proposition_and_variants (x y : ℝ) :
  -- Original proposition
  (x^2 + y^2 = 0 → x * y = 0) ∧
  -- Converse (false)
  ¬(x * y = 0 → x^2 + y^2 = 0) ∧
  -- Inverse (false)
  ¬(x^2 + y^2 ≠ 0 → x * y ≠ 0) ∧
  -- Contrapositive (true)
  (x * y ≠ 0 → x^2 + y^2 ≠ 0) := by
  sorry

end proposition_and_variants_l1887_188748


namespace red_balls_count_l1887_188798

/-- Given a jar with white and red balls where the ratio of white to red balls is 4:3 
    and there are 12 white balls, prove that there are 9 red balls. -/
theorem red_balls_count (white_balls : ℕ) (red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 4 / 3 → white_balls = 12 → red_balls = 9 := by
sorry

end red_balls_count_l1887_188798


namespace max_value_of_a_l1887_188784

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 12) :
  a ≤ (7 + Real.sqrt 46) / 3 :=
by sorry

end max_value_of_a_l1887_188784


namespace potato_division_l1887_188737

theorem potato_division (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) :
  total_potatoes = 24 →
  num_people = 3 →
  total_potatoes = num_people * potatoes_per_person →
  potatoes_per_person = 8 := by
  sorry

end potato_division_l1887_188737


namespace three_number_sum_l1887_188728

theorem three_number_sum : ∀ a b c : ℝ, 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 20 → 
  (a + b + c) / 3 = c - 30 → 
  a + b + c = 60 := by
  sorry

end three_number_sum_l1887_188728


namespace quadratic_factoring_l1887_188759

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α
  a_nonzero : a ≠ 0

/-- A factored form of a quadratic equation is a product of linear factors -/
structure FactoredForm (α : Type*) [Field α] where
  factor1 : α → α
  factor2 : α → α

/-- 
Given a quadratic equation that can be factored, 
it can be expressed as a multiplication of factors.
-/
theorem quadratic_factoring 
  {α : Type*} [Field α]
  (eq : QuadraticEquation α)
  (h_factorable : ∃ (f : FactoredForm α), 
    ∀ x, eq.a * x^2 + eq.b * x + eq.c = f.factor1 x * f.factor2 x) :
  ∃ (f : FactoredForm α), 
    ∀ x, eq.a * x^2 + eq.b * x + eq.c = f.factor1 x * f.factor2 x :=
by sorry

end quadratic_factoring_l1887_188759


namespace max_consecutive_sum_45_l1887_188762

/-- The sum of consecutive integers starting from a given integer -/
def sum_consecutive (start : ℤ) (count : ℕ) : ℤ :=
  (count : ℤ) * (2 * start + count - 1) / 2

/-- The property that a sequence of consecutive integers sums to 45 -/
def sums_to_45 (start : ℤ) (count : ℕ) : Prop :=
  sum_consecutive start count = 45

/-- The theorem stating that 90 is the maximum number of consecutive integers that sum to 45 -/
theorem max_consecutive_sum_45 :
  (∃ start : ℤ, sums_to_45 start 90) ∧
  (∀ count : ℕ, count > 90 → ∀ start : ℤ, ¬ sums_to_45 start count) :=
sorry

end max_consecutive_sum_45_l1887_188762


namespace fraction_evaluation_l1887_188781

theorem fraction_evaluation (a b : ℝ) (h : a ≠ b) : (a^4 - b^4) / (a^2 - b^2) = a^2 + b^2 := by
  sorry

end fraction_evaluation_l1887_188781


namespace smallest_square_containing_circle_l1887_188722

theorem smallest_square_containing_circle (r : ℝ) (h : r = 6) : 
  (2 * r) ^ 2 = 144 := by
  sorry

end smallest_square_containing_circle_l1887_188722


namespace regular_polygon_with_900_degree_sum_l1887_188788

theorem regular_polygon_with_900_degree_sum (n : ℕ) : 
  n > 2 → (n - 2) * 180 = 900 → n = 7 := by sorry

end regular_polygon_with_900_degree_sum_l1887_188788


namespace triangle_properties_l1887_188799

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a * Real.cos t.B + t.b * Real.cos t.A) / t.c = 2 * Real.cos t.C)
  (h2 : (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3)
  (h3 : t.a + t.b = 6) :
  t.C = π/3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry


end triangle_properties_l1887_188799


namespace son_age_l1887_188718

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 28 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry

end son_age_l1887_188718


namespace largest_y_floor_div_l1887_188714

theorem largest_y_floor_div : 
  ∀ y : ℝ, (↑(⌊y⌋) / y = 8 / 9) → y ≤ 63 / 8 := by
  sorry

end largest_y_floor_div_l1887_188714


namespace palindrome_existence_l1887_188779

/-- A number is a palindrome if it reads the same backwards and forwards in its decimal representation -/
def IsPalindrome (m : ℕ) : Prop :=
  ∃ (digits : List ℕ), m = digits.foldl (fun acc d => 10 * acc + d) 0 ∧ digits = digits.reverse

/-- For any natural number n, there exists a natural number N such that 9 * 5^n * N is a palindrome -/
theorem palindrome_existence (n : ℕ) : ∃ (N : ℕ), IsPalindrome (9 * 5^n * N) := by
  sorry

end palindrome_existence_l1887_188779


namespace shelter_dogs_l1887_188769

theorem shelter_dogs (x : ℕ) (dogs cats : ℕ) 
  (h1 : dogs * 7 = x * cats) 
  (h2 : dogs * 11 = 15 * (cats + 8)) : 
  dogs = 77 := by
  sorry

end shelter_dogs_l1887_188769


namespace arithmetic_geometric_sequence_property_l1887_188768

/-- An arithmetic sequence with non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_property
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_b7 : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry

end arithmetic_geometric_sequence_property_l1887_188768


namespace polynomial_coefficient_sums_l1887_188780

def P (x : ℝ) : ℝ := (2*x^2 - 2*x + 1)^17 * (3*x^2 - 3*x + 1)^17

theorem polynomial_coefficient_sums :
  (∀ x, P x = P 1) ∧
  (∀ x, (P x + P (-x)) / 2 = (1 + 35^17) / 2) ∧
  (∀ x, (P x - P (-x)) / 2 = (1 - 35^17) / 2) := by sorry

end polynomial_coefficient_sums_l1887_188780


namespace triangle_sin_B_l1887_188715

theorem triangle_sin_B (a b : ℝ) (A : ℝ) :
  a = Real.sqrt 6 →
  b = 2 →
  A = π / 4 →
  Real.sin (Real.arcsin ((b * Real.sin A) / a)) = Real.sqrt 3 / 3 :=
by sorry

end triangle_sin_B_l1887_188715


namespace condition_relationship_l1887_188720

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end condition_relationship_l1887_188720


namespace chinese_english_time_difference_l1887_188705

/-- The number of hours Ryan spends daily learning English -/
def english_hours : ℕ := 6

/-- The number of hours Ryan spends daily learning Chinese -/
def chinese_hours : ℕ := 7

/-- Theorem: The difference between the time spent on learning Chinese and English is 1 hour -/
theorem chinese_english_time_difference :
  chinese_hours - english_hours = 1 := by
  sorry

end chinese_english_time_difference_l1887_188705


namespace erased_number_problem_l1887_188702

theorem erased_number_problem (n : Nat) (x : Nat) : 
  n = 69 → 
  x ≤ n →
  x ≥ 1 →
  (((n * (n + 1)) / 2 - x) : ℚ) / (n - 1 : ℚ) = 35 + (7 : ℚ) / 17 →
  x = 7 := by
  sorry

end erased_number_problem_l1887_188702


namespace inequalities_not_always_hold_l1887_188767

theorem inequalities_not_always_hold :
  ∃ (a b c x y z : ℝ),
    x < a ∧ y < b ∧ z < c ∧
    ¬(x * y + y * z + z * x < a * b + b * c + c * a) ∧
    ¬(x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧
    ¬(x * y * z < a * b * c) :=
by sorry

end inequalities_not_always_hold_l1887_188767


namespace profit_percentage_60_percent_l1887_188783

/-- Profit percentage for 60% of apples given total apples, profit percentages, and sales distribution --/
theorem profit_percentage_60_percent (total_apples : ℝ) (profit_40_percent : ℝ) (total_profit_percent : ℝ) :
  total_apples = 280 →
  profit_40_percent = 10 →
  total_profit_percent = 22.000000000000007 →
  let profit_60_percent := 
    (total_profit_percent * total_apples - profit_40_percent * (0.4 * total_apples)) / (0.6 * total_apples) * 100
  profit_60_percent = 30 := by sorry

end profit_percentage_60_percent_l1887_188783
