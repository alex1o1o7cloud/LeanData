import Mathlib

namespace wage_period_theorem_l888_88888

/-- Represents the number of days a sum of money can pay wages -/
structure WagePeriod where
  b : ℕ  -- Days for B's wages
  c : ℕ  -- Days for C's wages
  both : ℕ  -- Days for both B and C's wages

/-- Given conditions on wage periods, proves the number of days both can be paid -/
theorem wage_period_theorem (w : WagePeriod) (hb : w.b = 12) (hc : w.c = 24) :
  w.both = 8 := by
  sorry

#check wage_period_theorem

end wage_period_theorem_l888_88888


namespace journey_matches_graph_characteristics_l888_88823

/-- Represents a point on the speed-time graph -/
structure SpeedTimePoint where
  time : ℝ
  speed : ℝ

/-- Represents a section of the speed-time graph -/
inductive GraphSection
  | Increasing : GraphSection
  | Flat : GraphSection
  | Decreasing : GraphSection

/-- Represents Mike's journey -/
structure Journey where
  cityTraffic : Bool
  highway : Bool
  workplace : Bool
  coffeeBreak : Bool
  workDuration : ℝ
  breakDuration : ℝ

/-- Defines the characteristics of the correct graph -/
def correctGraphCharacteristics : List GraphSection :=
  [GraphSection.Increasing, GraphSection.Flat, GraphSection.Increasing, 
   GraphSection.Flat, GraphSection.Decreasing]

/-- Theorem stating that Mike's journey matches the correct graph characteristics -/
theorem journey_matches_graph_characteristics (j : Journey) :
  j.cityTraffic = true →
  j.highway = true →
  j.workplace = true →
  j.coffeeBreak = true →
  j.workDuration = 2 →
  j.breakDuration = 0.5 →
  ∃ (graph : List GraphSection), graph = correctGraphCharacteristics := by
  sorry

#check journey_matches_graph_characteristics

end journey_matches_graph_characteristics_l888_88823


namespace negation_equivalence_l888_88804

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define the original proposition
def has_angle_le_60 (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i ≤ 60

-- Define the negation (assumption for proof by contradiction)
def all_angles_gt_60 (t : Triangle) : Prop :=
  ∀ i : Fin 3, t.angles i > 60

-- The theorem to prove
theorem negation_equivalence :
  ∀ t : Triangle, ¬(has_angle_le_60 t) ↔ all_angles_gt_60 t :=
sorry

end negation_equivalence_l888_88804


namespace new_person_weight_l888_88882

def initial_persons : ℕ := 6
def average_weight_increase : ℝ := 2
def replaced_person_weight : ℝ := 75

theorem new_person_weight :
  ∃ (new_weight : ℝ),
    new_weight = replaced_person_weight + initial_persons * average_weight_increase :=
by
  sorry

end new_person_weight_l888_88882


namespace initial_peanuts_count_l888_88824

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Mary adds to the box -/
def peanuts_added : ℕ := 12

/-- The final number of peanuts in the box after Mary adds more -/
def final_peanuts : ℕ := 16

/-- Theorem stating that the initial number of peanuts is 4 -/
theorem initial_peanuts_count : initial_peanuts = 4 := by
  sorry

end initial_peanuts_count_l888_88824


namespace magician_earnings_l888_88893

theorem magician_earnings 
  (price_per_deck : ℕ) 
  (initial_decks : ℕ) 
  (final_decks : ℕ) :
  price_per_deck = 2 →
  initial_decks = 5 →
  final_decks = 3 →
  (initial_decks - final_decks) * price_per_deck = 4 :=
by sorry

end magician_earnings_l888_88893


namespace gnome_count_l888_88838

/-- The number of garden gnomes with red hats, small noses, and striped shirts -/
def redHatSmallNoseStripedShirt (totalGnomes redHats bigNoses blueHatBigNoses : ℕ) : ℕ :=
  let blueHats := totalGnomes - redHats
  let smallNoses := totalGnomes - bigNoses
  let redHatSmallNoses := smallNoses - (blueHats - blueHatBigNoses)
  redHatSmallNoses / 2

/-- Theorem stating the number of garden gnomes with red hats, small noses, and striped shirts -/
theorem gnome_count : redHatSmallNoseStripedShirt 28 21 14 6 = 6 := by
  sorry

#eval redHatSmallNoseStripedShirt 28 21 14 6

end gnome_count_l888_88838


namespace alarm_clock_probability_l888_88866

theorem alarm_clock_probability (A B : ℝ) (hA : A = 0.80) (hB : B = 0.90) :
  1 - (1 - A) * (1 - B) = 0.98 := by
  sorry

end alarm_clock_probability_l888_88866


namespace infinitely_many_solutions_l888_88859

theorem infinitely_many_solutions (a : ℚ) : 
  (∀ x : ℚ, 4 * (3 * x - 2 * a) = 3 * (4 * x + 18)) ↔ a = -27/4 := by
  sorry

end infinitely_many_solutions_l888_88859


namespace triangular_prism_properties_l888_88889

/-- Represents a triangular prism -/
structure TriangularPrism where
  AB : ℝ
  AC : ℝ
  AA₁ : ℝ
  angleCAB : ℝ

/-- The volume of a triangular prism -/
def volume (p : TriangularPrism) : ℝ := sorry

/-- The surface area of a triangular prism -/
def surfaceArea (p : TriangularPrism) : ℝ := sorry

theorem triangular_prism_properties (p : TriangularPrism)
    (h1 : p.AB = 1)
    (h2 : p.AC = 1)
    (h3 : p.AA₁ = Real.sqrt 2)
    (h4 : p.angleCAB = 2 * π / 3) : -- 120° in radians
  volume p = Real.sqrt 6 / 4 ∧
  surfaceArea p = 2 * Real.sqrt 2 + Real.sqrt 6 + Real.sqrt 3 / 2 := by
  sorry

#check triangular_prism_properties

end triangular_prism_properties_l888_88889


namespace power_2_2013_mod_11_l888_88885

theorem power_2_2013_mod_11 : 2^2013 % 11 = 8 := by
  sorry

end power_2_2013_mod_11_l888_88885


namespace fuel_cost_per_liter_l888_88856

/-- The cost per liter of fuel given the tank capacity, initial fuel amount, and money spent. -/
theorem fuel_cost_per_liter
  (tank_capacity : ℝ)
  (initial_fuel : ℝ)
  (money_spent : ℝ)
  (h1 : tank_capacity = 150)
  (h2 : initial_fuel = 38)
  (h3 : money_spent = 336)
  : (money_spent / (tank_capacity - initial_fuel)) = 3 :=
by sorry

end fuel_cost_per_liter_l888_88856


namespace steps_to_school_l888_88803

/-- The number of steps Raine takes walking to and from school in five days -/
def total_steps : ℕ := 1500

/-- The number of days Raine walks to and from school -/
def days : ℕ := 5

/-- Proves that the number of steps Raine takes to walk to school is 150 -/
theorem steps_to_school : (total_steps / (2 * days) : ℕ) = 150 := by
  sorry

end steps_to_school_l888_88803


namespace pizza_order_l888_88864

theorem pizza_order (num_people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_people = 10)
  (h2 : slices_per_person = 2)
  (h3 : slices_per_pizza = 4) :
  (num_people * slices_per_person + slices_per_pizza - 1) / slices_per_pizza = 5 := by
  sorry

end pizza_order_l888_88864


namespace hyperbola_asymptotes_l888_88868

/-- Represents a hyperbola with equation x^2 - y^2/3 = 1 -/
def Hyperbola := {(x, y) : ℝ × ℝ | x^2 - y^2/3 = 1}

/-- The equation of asymptotes for the given hyperbola -/
def AsymptoteEquation (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- Theorem stating that the given equation represents the asymptotes of the hyperbola -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x, y) ∈ Hyperbola → (AsymptoteEquation x y ↔ (x, y) ∈ closure Hyperbola \ Hyperbola) :=
sorry

end hyperbola_asymptotes_l888_88868


namespace cylinder_surface_area_l888_88860

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches -/
theorem cylinder_surface_area : 
  ∀ (h r : ℝ), 
  h = 8 → 
  r = 3 → 
  2 * π * r * h + 2 * π * r^2 = 66 * π :=
by
  sorry

end cylinder_surface_area_l888_88860


namespace sum_of_ages_l888_88835

-- Define Rose's age
def rose_age : ℕ := 25

-- Define Rose's mother's age
def mother_age : ℕ := 75

-- Theorem: The sum of Rose's age and her mother's age is 100
theorem sum_of_ages : rose_age + mother_age = 100 := by
  sorry

end sum_of_ages_l888_88835


namespace cheese_division_theorem_l888_88816

theorem cheese_division_theorem (a b c d e f : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_order : a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f) :
  ∃ (S₁ S₂ : Finset ℝ), 
    S₁.card = 3 ∧ 
    S₂.card = 3 ∧ 
    S₁ ∩ S₂ = ∅ ∧ 
    S₁ ∪ S₂ = {a, b, c, d, e, f} ∧
    (S₁.sum id = S₂.sum id) :=
sorry

end cheese_division_theorem_l888_88816


namespace existence_of_100_pairs_l888_88812

def has_all_digits_at_least_6 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (n.digits 10) → d ≥ 6

theorem existence_of_100_pairs :
  ∃ S : Finset (ℕ × ℕ),
    S.card = 100 ∧
    (∀ (a b : ℕ), (a, b) ∈ S →
      has_all_digits_at_least_6 a ∧
      has_all_digits_at_least_6 b ∧
      has_all_digits_at_least_6 (a * b)) :=
sorry

end existence_of_100_pairs_l888_88812


namespace tenth_term_of_geometric_sequence_l888_88839

/-- Given a geometric sequence where the first term is 4 and the second term is 16/3,
    the 10th term of this sequence is 1048576/19683. -/
theorem tenth_term_of_geometric_sequence :
  let a₁ : ℚ := 4
  let a₂ : ℚ := 16/3
  let r : ℚ := a₂ / a₁
  let a₁₀ : ℚ := a₁ * r^9
  a₁₀ = 1048576/19683 := by sorry

end tenth_term_of_geometric_sequence_l888_88839


namespace range_of_a_l888_88850

theorem range_of_a (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 5 ∧ ∀ x ∈ s, (1 + a ≤ x ∧ x < 2)) → 
  -5 < a ∧ a ≤ -4 := by
  sorry

end range_of_a_l888_88850


namespace alissa_presents_l888_88815

theorem alissa_presents (ethan_presents : ℝ) (difference : ℝ) (alissa_presents : ℝ) : 
  ethan_presents = 31.0 → 
  difference = 22.0 → 
  alissa_presents = ethan_presents - difference → 
  alissa_presents = 9.0 :=
by sorry

end alissa_presents_l888_88815


namespace arithmetic_calculation_l888_88833

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end arithmetic_calculation_l888_88833


namespace school_election_votes_l888_88872

theorem school_election_votes (total_votes : ℕ) (brenda_votes : ℕ) : 
  brenda_votes = 50 → 
  4 * brenda_votes = total_votes →
  total_votes = 200 := by
sorry

end school_election_votes_l888_88872


namespace approx48000_accurate_to_thousand_l888_88886

/-- Represents an approximate value with its numerical value and accuracy -/
structure ApproximateValue where
  value : ℕ
  accuracy : ℕ

/-- Checks if the given approximate value is accurate to thousand -/
def isAccurateToThousand (av : ApproximateValue) : Prop :=
  av.accuracy = 1000

/-- The approximate value 48,000 -/
def approx48000 : ApproximateValue :=
  { value := 48000, accuracy := 1000 }

/-- Theorem stating that 48,000 is accurate to thousand -/
theorem approx48000_accurate_to_thousand :
  isAccurateToThousand approx48000 := by
  sorry

end approx48000_accurate_to_thousand_l888_88886


namespace michael_truck_rental_cost_l888_88837

/-- Calculates the total cost of renting a truck given the rental fee, charge per mile, and miles driven. -/
def truckRentalCost (rentalFee : ℚ) (chargePerMile : ℚ) (milesDriven : ℕ) : ℚ :=
  rentalFee + chargePerMile * milesDriven

/-- Proves that the total cost for Michael's truck rental is $95.74 -/
theorem michael_truck_rental_cost :
  truckRentalCost 20.99 0.25 299 = 95.74 := by
  sorry

end michael_truck_rental_cost_l888_88837


namespace simplify_radical_fraction_l888_88834

theorem simplify_radical_fraction (x : ℝ) (h1 : x < 0) :
  ((-x^3).sqrt / x) = -(-x).sqrt := by sorry

end simplify_radical_fraction_l888_88834


namespace square_area_after_cut_l888_88818

theorem square_area_after_cut (side : ℝ) (h1 : side > 0) : 
  side * (side - 3) = 40 → side * side = 64 := by sorry

end square_area_after_cut_l888_88818


namespace inequality_system_solution_l888_88855

theorem inequality_system_solution (x : ℝ) :
  (5 * x - 1 > 3 * (x + 1) ∧ x - 1 ≤ 7 - x) → (2 < x ∧ x ≤ 4) := by
  sorry

end inequality_system_solution_l888_88855


namespace a_share_is_one_third_l888_88879

/-- Represents the investment and profit distribution scenario -/
structure InvestmentScenario where
  initial_investment : ℝ
  annual_gain : ℝ
  months_in_year : ℕ

/-- Calculates the effective investment value for a partner -/
def effective_investment (scenario : InvestmentScenario) 
  (investment_multiplier : ℝ) (investment_duration : ℕ) : ℝ :=
  scenario.initial_investment * investment_multiplier * investment_duration

/-- Theorem stating that A's share of the gain is one-third of the total gain -/
theorem a_share_is_one_third (scenario : InvestmentScenario) 
  (h1 : scenario.months_in_year = 12)
  (h2 : scenario.annual_gain > 0) : 
  let a_investment := effective_investment scenario 1 scenario.months_in_year
  let b_investment := effective_investment scenario 2 6
  let c_investment := effective_investment scenario 3 4
  let total_effective_investment := a_investment + b_investment + c_investment
  scenario.annual_gain / 3 = (a_investment / total_effective_investment) * scenario.annual_gain := by
  sorry

#check a_share_is_one_third

end a_share_is_one_third_l888_88879


namespace existence_of_non_divisible_pair_l888_88875

theorem existence_of_non_divisible_pair (p : Nat) (h_prime : Prime p) (h_p_gt_3 : p > 3) :
  ∃ n : Nat, n > 0 ∧ n < p - 1 ∧
    ¬(p^2 ∣ n^(p-1) - 1) ∧ ¬(p^2 ∣ (n+1)^(p-1) - 1) := by
  sorry

end existence_of_non_divisible_pair_l888_88875


namespace theodore_wooden_statues_l888_88849

/-- Theodore's monthly statue production and earnings --/
structure StatueProduction where
  stone_statues : ℕ
  wooden_statues : ℕ
  stone_price : ℚ
  wooden_price : ℚ
  tax_rate : ℚ
  total_earnings_after_tax : ℚ

/-- Theorem: Theodore crafts 20 wooden statues per month --/
theorem theodore_wooden_statues (p : StatueProduction) 
  (h1 : p.stone_statues = 10)
  (h2 : p.stone_price = 20)
  (h3 : p.wooden_price = 5)
  (h4 : p.tax_rate = 1/10)
  (h5 : p.total_earnings_after_tax = 270) :
  p.wooden_statues = 20 := by
  sorry

#check theodore_wooden_statues

end theodore_wooden_statues_l888_88849


namespace chad_savings_l888_88858

/-- Chad's savings calculation --/
theorem chad_savings (savings_rate : ℚ) (mowing : ℚ) (birthday : ℚ) (video_games : ℚ) (odd_jobs : ℚ) : 
  savings_rate = 2/5 → 
  mowing = 600 → 
  birthday = 250 → 
  video_games = 150 → 
  odd_jobs = 150 → 
  savings_rate * (mowing + birthday + video_games + odd_jobs) = 460 := by
sorry

end chad_savings_l888_88858


namespace subtraction_to_perfect_square_l888_88802

theorem subtraction_to_perfect_square : ∃ n : ℕ, (92555 : ℕ) - 139 = n^2 := by
  sorry

end subtraction_to_perfect_square_l888_88802


namespace min_value_theorem_l888_88873

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : 2 = Real.sqrt (4^a * 2^b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 = Real.sqrt (4^x * 2^y) → 
    (2/a + 1/b) ≤ (2/x + 1/y) ∧ 
    (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2 = Real.sqrt (4^a₀ * 2^b₀) ∧ 2/a₀ + 1/b₀ = 9/2) :=
by sorry

end min_value_theorem_l888_88873


namespace unique_not_in_range_is_30_l888_88840

/-- Function f with the given properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem stating that 30 is the unique number not in the range of f -/
theorem unique_not_in_range_is_30
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : f a b c d 10 = 10)
  (h2 : f a b c d 50 = 50)
  (h3 : ∀ x ≠ -d/c, f a b c d (f a b c d x) = x) :
  ∃! y, ∀ x, f a b c d x ≠ y ∧ y = 30 :=
sorry

end unique_not_in_range_is_30_l888_88840


namespace x_equation_implies_zero_l888_88871

theorem x_equation_implies_zero (x : ℝ) (h : x + 1/x = Real.sqrt 5) :
  x^11 - 7*x^7 + x^3 = 0 := by
  sorry

end x_equation_implies_zero_l888_88871


namespace not_perfect_square_l888_88898

theorem not_perfect_square (n : ℕ) : ¬ ∃ (a : ℕ), 3 * n + 2 = a ^ 2 := by
  sorry

end not_perfect_square_l888_88898


namespace sum_of_ages_l888_88899

theorem sum_of_ages (a b c : ℕ) : 
  a = b + c + 16 → 
  a^2 = (b + c)^2 + 1632 → 
  a + b + c = 102 := by
sorry

end sum_of_ages_l888_88899


namespace min_value_quadratic_form_min_value_achievable_l888_88813

theorem min_value_quadratic_form (x y z : ℝ) :
  3 * x^2 + 2*x*y + 3 * y^2 + 2*y*z + 3 * z^2 - 3*x + 3*y - 3*z + 9 ≥ (3/2 : ℝ) :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), 3 * x^2 + 2*x*y + 3 * y^2 + 2*y*z + 3 * z^2 - 3*x + 3*y - 3*z + 9 = (3/2 : ℝ) :=
by sorry

end min_value_quadratic_form_min_value_achievable_l888_88813


namespace sum_min_max_value_l888_88828

theorem sum_min_max_value (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 30) : 
  let f := fun (x y z w v : ℝ) => 5 * (x^3 + y^3 + z^3 + w^3 + v^3) - (x^4 + y^4 + z^4 + w^4 + v^4)
  ∃ (m M : ℝ), 
    (∀ x y z w v, f x y z w v ≥ m) ∧ 
    (∃ x y z w v, f x y z w v = m) ∧
    (∀ x y z w v, f x y z w v ≤ M) ∧ 
    (∃ x y z w v, f x y z w v = M) ∧
    m + M = 94 :=
by sorry

end sum_min_max_value_l888_88828


namespace sum_in_base5_l888_88820

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of 201₅, 324₅, and 143₅ is equal to 1123₅ in base 5 --/
theorem sum_in_base5 :
  base10ToBase5 (base5ToBase10 201 + base5ToBase10 324 + base5ToBase10 143) = 1123 := by
  sorry

end sum_in_base5_l888_88820


namespace inequality_proof_l888_88807

theorem inequality_proof (x y : ℝ) (hx : |x| ≤ 1) (hy : |y| ≤ 1) : |x + y| ≤ |1 + x * y| := by
  sorry

end inequality_proof_l888_88807


namespace geometric_sequence_sum_l888_88892

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l888_88892


namespace quadratic_function_range_l888_88867

/-- A quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + m

/-- The range of f is [0, +∞) -/
def has_range_zero_to_infinity (m : ℝ) : Prop :=
  ∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f m x = y

theorem quadratic_function_range (m : ℝ) :
  has_range_zero_to_infinity m → m = 1 :=
by sorry

end quadratic_function_range_l888_88867


namespace equivalent_transitive_l888_88801

def IsGreat (f : ℕ → ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1) (n + 1) * f m n - f (m + 1) n * f m (n + 1) = 1

def Equivalent (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ → ℕ → ℤ, IsGreat f ∧ (∀ n : ℕ, f n 0 = A n ∧ f 0 n = B n)

theorem equivalent_transitive :
  ∀ A B C D : ℕ → ℤ,
    Equivalent A B → Equivalent B C → Equivalent C D → Equivalent D A :=
by sorry

end equivalent_transitive_l888_88801


namespace hawks_score_l888_88841

/-- The number of touchdowns scored by the Hawks -/
def touchdowns : ℕ := 3

/-- The number of points awarded for each touchdown -/
def points_per_touchdown : ℕ := 7

/-- The total points scored by the Hawks -/
def total_points : ℕ := touchdowns * points_per_touchdown

/-- Theorem stating that the total points scored by the Hawks is 21 -/
theorem hawks_score : total_points = 21 := by
  sorry

end hawks_score_l888_88841


namespace fraction_value_l888_88884

theorem fraction_value (p q : ℚ) (h : p / q = 7) : (p + q) / (p - q) = 4 / 3 := by
  sorry

end fraction_value_l888_88884


namespace high_speed_rail_distance_scientific_notation_l888_88869

theorem high_speed_rail_distance_scientific_notation :
  9280000000 = 9.28 * (10 ^ 9) := by
  sorry

end high_speed_rail_distance_scientific_notation_l888_88869


namespace jacobStatementsDisproved_l888_88822

-- Define the type for card sides
inductive CardSide
| Letter : Char → CardSide
| Number : Nat → CardSide

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define the properties of cards
def isVowel (c : Char) : Prop := c ∈ ['A', 'E', 'I', 'O', 'U']
def isEven (n : Nat) : Prop := n % 2 = 0
def isPrime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m > 1 → m < n → n % m ≠ 0)

-- Jacob's statements
def jacobStatement1 (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => isVowel c → isEven n
  | _ => True

def jacobStatement2 (card : Card) : Prop :=
  match card with
  | (CardSide.Number n, CardSide.Letter c) => isPrime n → isVowel c
  | _ => True

-- Define the set of cards
def cardSet : List Card := [
  (CardSide.Letter 'A', CardSide.Number 8),
  (CardSide.Letter 'R', CardSide.Number 5),
  (CardSide.Letter 'S', CardSide.Number 7),
  (CardSide.Number 1, CardSide.Letter 'R'),
  (CardSide.Number 8, CardSide.Letter 'S'),
  (CardSide.Number 5, CardSide.Letter 'A')
]

-- Theorem: There exist two cards that disprove at least one of Jacob's statements
theorem jacobStatementsDisproved : 
  ∃ (card1 card2 : Card), card1 ∈ cardSet ∧ card2 ∈ cardSet ∧ card1 ≠ card2 ∧
    (¬(jacobStatement1 card1) ∨ ¬(jacobStatement2 card1) ∨
     ¬(jacobStatement1 card2) ∨ ¬(jacobStatement2 card2)) :=
by sorry


end jacobStatementsDisproved_l888_88822


namespace max_integer_inequality_l888_88832

theorem max_integer_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2*a + b = 1) :
  ∀ m : ℤ, (∀ a b, a > 0 → b > 0 → 2*a + b = 1 → 2/a + 1/b ≥ m) → m ≤ 9 :=
sorry

end max_integer_inequality_l888_88832


namespace max_distance_complex_circle_l888_88896

theorem max_distance_complex_circle (z : ℂ) (z₀ : ℂ) :
  z₀ = 1 - 2*I →
  Complex.abs z = 3 →
  ∃ (max_dist : ℝ), max_dist = 3 + Real.sqrt 5 ∧
    ∀ (w : ℂ), Complex.abs w = 3 → Complex.abs (w - z₀) ≤ max_dist :=
by sorry

end max_distance_complex_circle_l888_88896


namespace polar_to_cartesian_l888_88845

theorem polar_to_cartesian (ρ θ x y : Real) :
  ρ * Real.cos θ = 1 ↔ x + y - 2 = 0 :=
sorry

end polar_to_cartesian_l888_88845


namespace inequality_equivalence_l888_88810

theorem inequality_equivalence (x : ℝ) :
  2 * |x - 2| - |x + 1| > 3 ↔ x < 0 ∨ x > 8 := by sorry

end inequality_equivalence_l888_88810


namespace sum_of_roots_squared_equation_l888_88865

theorem sum_of_roots_squared_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a + b = 14) := by
  sorry

end sum_of_roots_squared_equation_l888_88865


namespace tangent_slope_implies_abscissa_l888_88826

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem tangent_slope_implies_abscissa (x : ℝ) :
  (deriv f x = 3/2) → x = Real.log 2 := by
  sorry

end tangent_slope_implies_abscissa_l888_88826


namespace puppy_cost_proof_l888_88891

/-- Given a purchase of puppies with specific conditions, prove the cost of non-sale puppies. -/
theorem puppy_cost_proof (total_cost : ℕ) (sale_price : ℕ) (num_puppies : ℕ) (num_sale_puppies : ℕ) :
  total_cost = 800 →
  sale_price = 150 →
  num_puppies = 5 →
  num_sale_puppies = 3 →
  ∃ (non_sale_price : ℕ), 
    non_sale_price * (num_puppies - num_sale_puppies) + sale_price * num_sale_puppies = total_cost ∧
    non_sale_price = 175 := by
  sorry

end puppy_cost_proof_l888_88891


namespace fraction_equality_l888_88853

theorem fraction_equality (p q : ℝ) (h : p / q - q / p = 21 / 10) :
  4 * p / q + 4 * q / p = 16.8 := by
  sorry

end fraction_equality_l888_88853


namespace division_remainder_proof_l888_88821

theorem division_remainder_proof :
  ∀ (dividend quotient divisor remainder : ℕ),
    dividend = 144 →
    quotient = 13 →
    divisor = 11 →
    dividend = divisor * quotient + remainder →
    remainder = 1 := by
  sorry

end division_remainder_proof_l888_88821


namespace max_weight_proof_l888_88851

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 5

/-- The minimum weight of each crate in kilograms -/
def min_crate_weight : ℕ := 150

/-- The maximum weight of crates on a single trip in kilograms -/
def max_total_weight : ℕ := max_crates * min_crate_weight

theorem max_weight_proof :
  max_total_weight = 750 := by
  sorry

end max_weight_proof_l888_88851


namespace twenty_squares_in_four_by_five_grid_l888_88814

/-- Represents a grid of points -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Counts the number of squares of a given size in a grid -/
def countSquares (g : Grid) (size : Nat) : Nat :=
  (g.rows - size + 1) * (g.cols - size + 1)

/-- The total number of squares in a grid -/
def totalSquares (g : Grid) : Nat :=
  countSquares g 1 + countSquares g 2 + countSquares g 3

/-- Theorem: In a 4x5 grid, the total number of squares is 20 -/
theorem twenty_squares_in_four_by_five_grid :
  totalSquares ⟨4, 5⟩ = 20 := by
  sorry

#eval totalSquares ⟨4, 5⟩

end twenty_squares_in_four_by_five_grid_l888_88814


namespace cubic_root_ratio_l888_88800

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 4 ∨ x = 5 ∨ x = 6) :
  c / d = 1 / 8 := by
  sorry

end cubic_root_ratio_l888_88800


namespace smallest_n_for_Bn_radius_greater_than_two_l888_88876

theorem smallest_n_for_Bn_radius_greater_than_two :
  (∃ n : ℕ+, (∀ k : ℕ+, k < n → Real.sqrt k - 1 ≤ 2) ∧ Real.sqrt n - 1 > 2) ∧
  (∀ n : ℕ+, (∀ k : ℕ+, k < n → Real.sqrt k - 1 ≤ 2) ∧ Real.sqrt n - 1 > 2 → n = 10) := by
  sorry

end smallest_n_for_Bn_radius_greater_than_two_l888_88876


namespace b_minus_d_squared_l888_88819

theorem b_minus_d_squared (a b c d e : ℝ) 
  (eq1 : a - b - c + d = 13)
  (eq2 : a + b - c - d = 9)
  (eq3 : a - b + c + e = 11) : 
  (b - d)^2 = 4 := by
  sorry

end b_minus_d_squared_l888_88819


namespace hawks_touchdowns_l888_88844

theorem hawks_touchdowns (total_points : ℕ) (points_per_touchdown : ℕ) 
  (h1 : total_points = 21) 
  (h2 : points_per_touchdown = 7) : 
  total_points / points_per_touchdown = 3 := by
  sorry

end hawks_touchdowns_l888_88844


namespace tangent_perpendicular_theorem_l888_88862

noncomputable def f (x : ℝ) : ℝ := x^4

def perpendicular_line (x y : ℝ) : Prop := x + 4*y - 8 = 0

def tangent_line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

theorem tangent_perpendicular_theorem :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = f x₀ ∧ 
    (∃ (a b c : ℝ), tangent_line a b c x₀ y₀ ∧ 
      (∀ (x y : ℝ), perpendicular_line x y → 
        (a*1 + b*4 = 0))) → 
    (∃ (x y : ℝ), tangent_line 4 (-1) (-3) x y) :=
sorry

end tangent_perpendicular_theorem_l888_88862


namespace integral_x4_over_2minusx2_32_l888_88890

theorem integral_x4_over_2minusx2_32 :
  ∫ x in (0:ℝ)..1, x^4 / (2 - x^2)^(3/2) = 5/2 - 3*π/4 := by
  sorry

end integral_x4_over_2minusx2_32_l888_88890


namespace divisor_remainders_l888_88877

theorem divisor_remainders (n : ℕ) 
  (h : ∀ i ∈ Finset.range 1012, ∃ (d : ℕ), d ∣ n ∧ d % 2013 = 1001 + i) :
  ∀ k ∈ Finset.range 2012, ∃ (d : ℕ), d ∣ n^2 ∧ d % 2013 = k + 1 := by
sorry

end divisor_remainders_l888_88877


namespace copresidents_count_l888_88878

/-- Represents a club with members distributed across departments. -/
structure Club where
  total_members : ℕ
  num_departments : ℕ
  members_per_department : ℕ
  h_total : total_members = num_departments * members_per_department

/-- The number of ways to choose co-presidents from different departments. -/
def choose_copresidents (c : Club) : ℕ :=
  (c.num_departments * c.members_per_department * (c.num_departments - 1) * c.members_per_department) / 2

/-- Theorem stating the number of ways to choose co-presidents for the given club configuration. -/
theorem copresidents_count (c : Club) 
  (h_total : c.total_members = 24)
  (h_departments : c.num_departments = 4)
  (h_distribution : c.members_per_department = 6) : 
  choose_copresidents c = 54 := by
  sorry

#eval choose_copresidents ⟨24, 4, 6, rfl⟩

end copresidents_count_l888_88878


namespace like_terms_exponents_l888_88830

theorem like_terms_exponents (a b : ℝ) (x y : ℤ) : 
  (∃ (k : ℝ), -4 * a^(x-y) * b^4 = k * a^2 * b^(x+y)) → 
  (x = 3 ∧ y = 1) :=
by sorry

end like_terms_exponents_l888_88830


namespace x_value_proof_l888_88874

theorem x_value_proof : 
  ∀ x : ℝ, x = 143 * (1 + 32.5 / 100) → x = 189.475 := by
  sorry

end x_value_proof_l888_88874


namespace f_1384_bounds_l888_88881

/-- An n-mino is a shape made up of n equal squares connected edge-to-edge. -/
def Mino (n : ℕ) : Type := Unit  -- We don't need to define the full structure for this proof

/-- f(n) is the least number such that there exists an f(n)-mino containing every n-mino -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds for f(1384) -/
theorem f_1384_bounds : 10000 ≤ f 1384 ∧ f 1384 ≤ 960000 := by sorry

end f_1384_bounds_l888_88881


namespace person_y_speed_l888_88809

-- Define the river and docks
structure River :=
  (current_speed : ℝ)

structure Dock :=
  (position : ℝ)

-- Define the persons and their boats
structure Person :=
  (rowing_speed : ℝ)
  (starting_dock : Dock)

-- Define the scenario
def Scenario (river : River) (x y : Person) :=
  (x.rowing_speed = 6) ∧ 
  (x.starting_dock.position < y.starting_dock.position) ∧
  (∃ t : ℝ, t > 0 ∧ t * (x.rowing_speed - river.current_speed) = t * (y.rowing_speed + river.current_speed)) ∧
  (∃ t : ℝ, t > 0 ∧ t * (y.rowing_speed + river.current_speed) = t * (x.rowing_speed + river.current_speed) + 4 * (y.rowing_speed - x.rowing_speed)) ∧
  (4 * (x.rowing_speed - river.current_speed + y.rowing_speed + river.current_speed) = 16 * (y.rowing_speed - x.rowing_speed))

-- Theorem statement
theorem person_y_speed (river : River) (x y : Person) 
  (h : Scenario river x y) : y.rowing_speed = 10 :=
sorry

end person_y_speed_l888_88809


namespace jessica_exam_time_l888_88880

/-- Calculates the remaining time for Jessica to finish her exam -/
def remaining_time (total_time minutes_used questions_total questions_answered : ℕ) : ℕ :=
  total_time - minutes_used

/-- Proves that Jessica will have 48 minutes left when she finishes the exam -/
theorem jessica_exam_time : remaining_time 60 12 80 16 = 48 := by
  sorry

end jessica_exam_time_l888_88880


namespace distance_between_X_and_Y_l888_88817

/-- The distance between points X and Y in miles -/
def D : ℝ := sorry

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 5

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 6

/-- The time in hours that Bob walks before meeting Yolanda -/
def bob_time : ℝ := sorry

/-- The distance Bob walks before meeting Yolanda in miles -/
def bob_distance : ℝ := 30

theorem distance_between_X_and_Y : D = 60 := by
  sorry

end distance_between_X_and_Y_l888_88817


namespace rectangle_y_value_l888_88848

/-- A rectangle with vertices at (1, y), (9, y), (1, 5), and (9, 5), where y is positive and the area is 64 square units, has y = 13. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) (h2 : (9 - 1) * (y - 5) = 64) : y = 13 := by
  sorry

end rectangle_y_value_l888_88848


namespace expression_simplification_l888_88895

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x + a)^4 / ((a - b) * (a - c)) + (x + b)^4 / ((b - a) * (b - c)) + (x + c)^4 / ((c - a) * (c - b)) =
  a + b + c + 3 * x^2 := by
  sorry

end expression_simplification_l888_88895


namespace cube_cut_forms_regular_hexagons_l888_88808

-- Define a cube
structure Cube where
  side : ℝ
  side_positive : side > 0

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a regular hexagon
structure RegularHexagon where
  side : ℝ
  side_positive : side > 0

-- Function to get midpoints of cube edges
def getMidpoints (c : Cube) : List Point3D :=
  sorry

-- Function to define a plane through midpoints
def planeThroughMidpoints (midpoints : List Point3D) : Plane3D :=
  sorry

-- Function to determine if a plane intersects a cube to form regular hexagons
def intersectionFormsRegularHexagons (c : Cube) (p : Plane3D) : Prop :=
  sorry

-- Theorem statement
theorem cube_cut_forms_regular_hexagons (c : Cube) :
  let midpoints := getMidpoints c
  let cuttingPlane := planeThroughMidpoints midpoints
  intersectionFormsRegularHexagons c cuttingPlane :=
sorry

end cube_cut_forms_regular_hexagons_l888_88808


namespace inequality_solutions_l888_88847

theorem inequality_solutions :
  -- Part 1
  (∀ x : ℝ, (3*x - 2)/(x - 1) > 1 ↔ (x > 1 ∨ x < 1/2)) ∧
  -- Part 2
  (∀ a x : ℝ, 
    (a = 0 → x^2 - a*x - 2*a^2 < 0 ↔ False) ∧
    (a > 0 → (x^2 - a*x - 2*a^2 < 0 ↔ -a < x ∧ x < 2*a)) ∧
    (a < 0 → (x^2 - a*x - 2*a^2 < 0 ↔ 2*a < x ∧ x < -a))) :=
by sorry

end inequality_solutions_l888_88847


namespace quaternary_123_equals_27_l888_88842

/-- Converts a quaternary (base-4) digit to its decimal value --/
def quaternary_to_decimal (digit : Nat) : Nat :=
  if digit < 4 then digit else 0

/-- Represents the quaternary number 123 --/
def quaternary_123 : List Nat := [1, 2, 3]

/-- Converts a list of quaternary digits to its decimal value --/
def quaternary_list_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + quaternary_to_decimal d * (4 ^ i)) 0

theorem quaternary_123_equals_27 :
  quaternary_list_to_decimal quaternary_123 = 27 := by
  sorry

end quaternary_123_equals_27_l888_88842


namespace stating_exist_same_arrangement_l888_88831

/-- The size of the grid -/
def grid_size : Nat := 25

/-- The size of the sub-squares we're considering -/
def square_size : Nat := 3

/-- The number of possible 3x3 squares in a 25x25 grid -/
def num_squares : Nat := (grid_size - square_size + 1) ^ 2

/-- The number of possible arrangements of plus signs in a 3x3 square -/
def num_arrangements : Nat := 2 ^ (square_size ^ 2)

/-- 
Theorem stating that there exist at least two 3x3 squares 
with the same arrangement of plus signs in a 25x25 grid 
-/
theorem exist_same_arrangement : num_squares > num_arrangements := by sorry

end stating_exist_same_arrangement_l888_88831


namespace store_inventory_difference_l888_88836

theorem store_inventory_difference : 
  ∀ (apples regular_soda diet_soda : ℕ),
    apples = 36 →
    regular_soda = 80 →
    diet_soda = 54 →
    regular_soda + diet_soda - apples = 98 :=
by
  sorry

end store_inventory_difference_l888_88836


namespace triangle_height_problem_l888_88852

theorem triangle_height_problem (base1 height1 base2 : ℝ) 
  (h_base1 : base1 = 15)
  (h_height1 : height1 = 12)
  (h_base2 : base2 = 20)
  (h_area_relation : base2 * (base1 * height1) = 2 * base1 * (base2 * height1)) :
  ∃ height2 : ℝ, height2 = 18 ∧ base2 * height2 = 2 * (base1 * height1) := by
sorry

end triangle_height_problem_l888_88852


namespace smallest_prime_with_digit_sum_25_l888_88883

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem smallest_prime_with_digit_sum_25 :
  ∃ p : ℕ, is_prime p ∧ digit_sum p = 25 ∧
  ∀ q : ℕ, is_prime q ∧ digit_sum q = 25 → p ≤ q :=
by sorry

end smallest_prime_with_digit_sum_25_l888_88883


namespace partial_fraction_sum_zero_l888_88825

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5) →
  A + B + C + D + E = 0 := by
sorry

end partial_fraction_sum_zero_l888_88825


namespace determinant_of_specific_matrix_l888_88827

theorem determinant_of_specific_matrix : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 4; 0, 6, -2; 5, -3, 2]
  Matrix.det A = -68 := by
  sorry

end determinant_of_specific_matrix_l888_88827


namespace replacement_concentration_theorem_l888_88811

/-- Represents a hydrochloric acid solution --/
structure HClSolution where
  total_mass : ℝ
  concentration : ℝ

/-- Calculates the mass of pure HCl in a solution --/
def pure_hcl_mass (solution : HClSolution) : ℝ :=
  solution.total_mass * solution.concentration

theorem replacement_concentration_theorem 
  (initial_solution : HClSolution)
  (drained_mass : ℝ)
  (final_solution : HClSolution)
  (replacement_solution : HClSolution)
  (h1 : initial_solution.total_mass = 300)
  (h2 : initial_solution.concentration = 0.2)
  (h3 : drained_mass = 25)
  (h4 : final_solution.total_mass = initial_solution.total_mass)
  (h5 : final_solution.concentration = 0.25)
  (h6 : replacement_solution.total_mass = drained_mass)
  (h7 : pure_hcl_mass final_solution = 
        pure_hcl_mass initial_solution - pure_hcl_mass replacement_solution + 
        pure_hcl_mass replacement_solution) :
  replacement_solution.concentration = 0.8 := by
  sorry

#check replacement_concentration_theorem

end replacement_concentration_theorem_l888_88811


namespace two_digit_number_puzzle_l888_88857

/-- Given a two-digit number with digit sum 6, if the product of this number and
    the number formed by swapping its digits is 1008, then the original number
    is either 42 or 24. -/
theorem two_digit_number_puzzle (n : ℕ) : 
  (n ≥ 10 ∧ n < 100) →  -- n is a two-digit number
  (n / 10 + n % 10 = 6) →  -- digit sum is 6
  (n * (10 * (n % 10) + (n / 10)) = 1008) →  -- product condition
  (n = 42 ∨ n = 24) := by
sorry

end two_digit_number_puzzle_l888_88857


namespace sqrt_neg_2x_cubed_eq_neg_x_sqrt_neg_2x_l888_88894

theorem sqrt_neg_2x_cubed_eq_neg_x_sqrt_neg_2x :
  ∀ x : ℝ, x ≤ 0 → Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) :=
by sorry

end sqrt_neg_2x_cubed_eq_neg_x_sqrt_neg_2x_l888_88894


namespace special_octagon_regions_l888_88854

/-- Represents an octagon with specific properties -/
structure SpecialOctagon where
  angles : Fin 8 → ℝ
  sides : Fin 8 → ℝ
  all_angles_135 : ∀ i, angles i = 135
  alternating_sides : ∀ i, sides i = if i % 2 = 0 then 1 else Real.sqrt 2

/-- Counts the regions formed by drawing all sides and diagonals of the octagon -/
def count_regions (o : SpecialOctagon) : ℕ :=
  84

/-- Theorem stating that the special octagon is divided into 84 regions -/
theorem special_octagon_regions (o : SpecialOctagon) : 
  count_regions o = 84 := by sorry

end special_octagon_regions_l888_88854


namespace pages_read_relationship_l888_88843

/-- Represents the number of pages read on each night --/
structure PagesRead where
  night1 : ℕ
  night2 : ℕ
  night3 : ℕ

/-- Theorem stating the relationship between pages read on night 3 and the other nights --/
theorem pages_read_relationship (p : PagesRead) (total : ℕ) : 
  p.night1 = 30 →
  p.night2 = 2 * p.night1 - 2 →
  total = p.night1 + p.night2 + p.night3 →
  total = 179 →
  p.night3 = total - (p.night1 + p.night2) := by
  sorry

end pages_read_relationship_l888_88843


namespace grid_toothpicks_l888_88806

/-- Calculates the total number of toothpicks in a grid with diagonals -/
def total_toothpicks (length width : ℕ) : ℕ :=
  let vertical := (length + 1) * width
  let horizontal := (width + 1) * length
  let diagonal := 2 * (length * width)
  vertical + horizontal + diagonal

/-- Theorem stating that a 50x20 grid with diagonals uses 4070 toothpicks -/
theorem grid_toothpicks : total_toothpicks 50 20 = 4070 := by
  sorry

end grid_toothpicks_l888_88806


namespace max_sum_of_factors_l888_88863

theorem max_sum_of_factors (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 →
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2023 →
  A + B + C ≤ 297 :=
by sorry

end max_sum_of_factors_l888_88863


namespace max_value_abc_l888_88829

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  ∃ (max : ℝ), max = 8 ∧ ∀ (x : ℝ), x = a + b^2 + c^3 → x ≤ max :=
sorry

end max_value_abc_l888_88829


namespace smallest_satisfying_polygon_l888_88887

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def satisfies_conditions (n : ℕ) : Prop :=
  (number_of_diagonals n) * 4 = n * 7 ∧
  (number_of_diagonals n + n) % 2 = 0 ∧
  number_of_diagonals n + n > 50

theorem smallest_satisfying_polygon : 
  satisfies_conditions 12 ∧ 
  ∀ m : ℕ, m < 12 → ¬(satisfies_conditions m) :=
sorry

end smallest_satisfying_polygon_l888_88887


namespace sample_size_is_twenty_l888_88897

/-- Represents the number of brands for each dairy product type -/
structure DairyBrands where
  pureMilk : ℕ
  yogurt : ℕ
  infantFormula : ℕ
  adultFormula : ℕ

/-- Represents the sample sizes for each dairy product type -/
structure SampleSizes where
  pureMilk : ℕ
  yogurt : ℕ
  infantFormula : ℕ
  adultFormula : ℕ

/-- Calculates the total sample size given the sample sizes for each product type -/
def totalSampleSize (s : SampleSizes) : ℕ :=
  s.pureMilk + s.yogurt + s.infantFormula + s.adultFormula

/-- Theorem stating that the total sample size is 20 given the problem conditions -/
theorem sample_size_is_twenty (brands : DairyBrands)
    (h1 : brands.pureMilk = 30)
    (h2 : brands.yogurt = 10)
    (h3 : brands.infantFormula = 35)
    (h4 : brands.adultFormula = 25)
    (sample : SampleSizes)
    (h5 : sample.infantFormula = 7)
    (h6 : sample.pureMilk * brands.infantFormula = brands.pureMilk * sample.infantFormula)
    (h7 : sample.yogurt * brands.infantFormula = brands.yogurt * sample.infantFormula)
    (h8 : sample.adultFormula * brands.infantFormula = brands.adultFormula * sample.infantFormula) :
  totalSampleSize sample = 20 := by
  sorry


end sample_size_is_twenty_l888_88897


namespace systematic_sampling_interval_l888_88861

/-- The interval of segmentation for systematic sampling -/
def intervalOfSegmentation (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The interval of segmentation for a population of 2000 and sample size of 40 is 50 -/
theorem systematic_sampling_interval :
  intervalOfSegmentation 2000 40 = 50 := by
  sorry

end systematic_sampling_interval_l888_88861


namespace no_such_function_exists_l888_88846

theorem no_such_function_exists : ¬∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y := by
  sorry

end no_such_function_exists_l888_88846


namespace max_coconuts_count_l888_88805

/-- Represents the trading ratios and final goat count -/
structure TradingSystem where
  coconuts_per_crab : ℕ
  crabs_per_goat : ℕ
  final_goats : ℕ

/-- Calculates the number of coconuts Max has -/
def coconuts_count (ts : TradingSystem) : ℕ :=
  ts.coconuts_per_crab * ts.crabs_per_goat * ts.final_goats

/-- Theorem stating that Max has 342 coconuts given the trading system -/
theorem max_coconuts_count :
  let ts : TradingSystem := ⟨3, 6, 19⟩
  coconuts_count ts = 342 := by
  sorry


end max_coconuts_count_l888_88805


namespace percentage_reading_two_novels_l888_88870

theorem percentage_reading_two_novels
  (total_students : ℕ)
  (three_or_more : ℚ)
  (one_novel : ℚ)
  (no_novels : ℕ)
  (h1 : total_students = 240)
  (h2 : three_or_more = 1 / 6)
  (h3 : one_novel = 5 / 12)
  (h4 : no_novels = 16) :
  (total_students - (three_or_more * total_students).num - (one_novel * total_students).num - no_novels : ℚ) / total_students * 100 = 35 := by
sorry


end percentage_reading_two_novels_l888_88870
