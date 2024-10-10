import Mathlib

namespace smallest_w_l3117_311781

theorem smallest_w (w : ℕ+) : 
  (∃ k : ℕ, 936 * w.val = k * 2^5) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 3^3) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 11^2) →
  w.val ≥ 4356 :=
by sorry

end smallest_w_l3117_311781


namespace tickets_purchased_l3117_311784

theorem tickets_purchased (olivia_money : ℕ) (nigel_money : ℕ) (ticket_cost : ℕ) (money_left : ℕ) :
  olivia_money = 112 →
  nigel_money = 139 →
  ticket_cost = 28 →
  money_left = 83 →
  (olivia_money + nigel_money - money_left) / ticket_cost = 6 :=
by sorry

end tickets_purchased_l3117_311784


namespace cos_300_degrees_l3117_311768

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l3117_311768


namespace three_valid_plans_l3117_311713

/-- Represents the cost and construction details of parking spaces -/
structure ParkingProject where
  aboveGroundCost : ℚ
  undergroundCost : ℚ
  totalSpaces : ℕ
  minInvestment : ℚ
  maxInvestment : ℚ

/-- Calculates the number of valid construction plans -/
def validConstructionPlans (project : ParkingProject) : ℕ :=
  (project.totalSpaces + 1).fold
    (λ count aboveGround =>
      let underground := project.totalSpaces - aboveGround
      let cost := project.aboveGroundCost * aboveGround + project.undergroundCost * underground
      if project.minInvestment < cost ∧ cost ≤ project.maxInvestment then
        count + 1
      else
        count)
    0

/-- Theorem stating that there are exactly 3 valid construction plans -/
theorem three_valid_plans (project : ParkingProject)
  (h1 : project.aboveGroundCost + project.undergroundCost = 0.6)
  (h2 : 3 * project.aboveGroundCost + 2 * project.undergroundCost = 1.3)
  (h3 : project.totalSpaces = 50)
  (h4 : project.minInvestment = 12)
  (h5 : project.maxInvestment = 13) :
  validConstructionPlans project = 3 := by
  sorry

#eval validConstructionPlans {
  aboveGroundCost := 0.1,
  undergroundCost := 0.5,
  totalSpaces := 50,
  minInvestment := 12,
  maxInvestment := 13
}

end three_valid_plans_l3117_311713


namespace village_population_l3117_311771

/-- If 40% of a population is 23040, then the total population is 57600. -/
theorem village_population (population : ℕ) : (40 : ℕ) * population / 100 = 23040 → population = 57600 := by
  sorry

end village_population_l3117_311771


namespace joan_quarters_l3117_311734

def total_cents : ℕ := 150
def cents_per_quarter : ℕ := 25

theorem joan_quarters : total_cents / cents_per_quarter = 6 := by
  sorry

end joan_quarters_l3117_311734


namespace car_distances_l3117_311797

/-- Represents the possible distances between two cars after one hour, given their initial distance and speeds. -/
def possible_distances (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : Set ℝ :=
  { d | ∃ (direction1 direction2 : Bool),
      d = |initial_distance + (if direction1 then speed1 else -speed1) - (if direction2 then speed2 else -speed2)| }

/-- Theorem stating the possible distances between two cars after one hour. -/
theorem car_distances (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ)
    (h_initial : initial_distance = 200)
    (h_speed1 : speed1 = 60)
    (h_speed2 : speed2 = 80) :
    possible_distances initial_distance speed1 speed2 = {60, 340, 180, 220} := by
  sorry

end car_distances_l3117_311797


namespace probability_less_equal_nine_l3117_311717

def card_set : Finset ℕ := {1, 3, 4, 6, 7, 9}

theorem probability_less_equal_nine : 
  (card_set.filter (λ x => x ≤ 9)).card / card_set.card = 1 := by
  sorry

end probability_less_equal_nine_l3117_311717


namespace read_book_series_l3117_311773

/-- The number of weeks required to read a book series -/
def weeks_to_read (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  let remaining_books := total_books - first_week - second_week
  let additional_weeks := (remaining_books + subsequent_weeks - 1) / subsequent_weeks
  2 + additional_weeks

/-- Theorem: It takes 7 weeks to read the book series under given conditions -/
theorem read_book_series : weeks_to_read 54 6 3 9 = 7 := by
  sorry

end read_book_series_l3117_311773


namespace expression_evaluation_l3117_311733

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := -1
  3 * (2 * a^2 * b - a * b^2) - 2 * (5 * a^2 * b - 2 * a * b^2) = 18 := by
  sorry

end expression_evaluation_l3117_311733


namespace tiaorizhi_approximation_of_pi_l3117_311769

def tiaorizhi (a b c d : ℕ) : ℚ := (b + d) / (a + c)

theorem tiaorizhi_approximation_of_pi :
  let initial_lower : ℚ := 3 / 1
  let initial_upper : ℚ := 7 / 2
  let step1 : ℚ := tiaorizhi 1 3 2 7
  let step2 : ℚ := tiaorizhi 1 3 4 13
  let step3 : ℚ := tiaorizhi 1 3 5 16
  initial_lower < Real.pi ∧ Real.pi < initial_upper →
  step3 - Real.pi < 0.1 ∧ Real.pi < step3 := by
  sorry

end tiaorizhi_approximation_of_pi_l3117_311769


namespace bobby_total_blocks_l3117_311788

def bobby_blocks : ℕ := 2
def father_gift : ℕ := 6

theorem bobby_total_blocks :
  bobby_blocks + father_gift = 8 := by
  sorry

end bobby_total_blocks_l3117_311788


namespace prob_three_even_out_of_five_l3117_311742

-- Define a fair 6-sided die
def FairDie := Fin 6

-- Define the probability of rolling an even number on a single die
def probEven : ℚ := 1 / 2

-- Define the number of dice
def numDice : ℕ := 5

-- Define the number of dice we want to show even
def numEven : ℕ := 3

-- Theorem statement
theorem prob_three_even_out_of_five :
  (Nat.choose numDice numEven : ℚ) * probEven ^ numDice = 5 / 16 := by
  sorry

end prob_three_even_out_of_five_l3117_311742


namespace train_bridge_crossing_time_l3117_311761

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 100) 
  (h2 : bridge_length = 170) 
  (h3 : train_speed_kmph = 36) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 27 := by
  sorry

#check train_bridge_crossing_time

end train_bridge_crossing_time_l3117_311761


namespace no_zeros_in_interval_l3117_311764

open Real

theorem no_zeros_in_interval (ω : ℝ) (h_ω_pos : ω > 0) :
  (∀ x ∈ Set.Ioo (π / 2) (3 * π / 2), cos (ω * x - 5 * π / 6) ≠ 0) →
  ω ∈ Set.Ioc 0 (2 / 9) ∪ Set.Icc (2 / 3) (8 / 9) :=
by sorry

end no_zeros_in_interval_l3117_311764


namespace concert_problem_l3117_311728

/-- Represents the number of songs sung by each friend -/
structure SongCount where
  lucy : ℕ
  gina : ℕ
  zoe : ℕ
  sara : ℕ

/-- Calculates the total number of songs performed by the trios -/
def totalSongs (sc : SongCount) : ℚ :=
  (sc.lucy + sc.gina + sc.zoe + sc.sara) / 3

/-- Represents the conditions of the problem -/
def validSongCount (sc : SongCount) : Prop :=
  sc.sara = 9 ∧
  sc.lucy = 3 ∧
  sc.zoe = sc.sara ∧
  sc.gina > sc.lucy ∧
  sc.gina ≤ sc.sara ∧
  (sc.lucy + sc.gina) % 4 = 0

theorem concert_problem (sc : SongCount) (h : validSongCount sc) :
  totalSongs sc = 9 ∨ totalSongs sc = 10 := by
  sorry


end concert_problem_l3117_311728


namespace system_solutions_l3117_311747

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ) :=
  {(0, 0), (2 + Real.sqrt 2, 2 + Real.sqrt 2), (2 - Real.sqrt 2, 2 - Real.sqrt 2)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ (x y : ℝ), system x y ↔ (x, y) ∈ solutions :=
sorry

end system_solutions_l3117_311747


namespace vector_operation_l3117_311704

/-- Given vectors a and b in R², prove that 2a - b equals (-1, 0) --/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (3, 4)) :
  2 • a - b = (-1, 0) := by
  sorry

end vector_operation_l3117_311704


namespace probability_no_more_than_five_girls_between_first_last_boys_l3117_311794

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

def valid_arrangements (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_no_more_than_five_girls_between_first_last_boys :
  (valid_arrangements 14 9 + 6 * valid_arrangements 13 8) / valid_arrangements total_children num_boys =
  (valid_arrangements 14 9 + 6 * valid_arrangements 13 8) / valid_arrangements 20 9 :=
by sorry

end probability_no_more_than_five_girls_between_first_last_boys_l3117_311794


namespace logical_propositions_l3117_311798

theorem logical_propositions (p q : Prop) : 
  (((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q))) ∧
  (((¬p) → ¬(p ∨ q)) ∧ ¬((p ∨ q) → ¬(¬p))) := by
  sorry

end logical_propositions_l3117_311798


namespace division_of_monomials_l3117_311727

theorem division_of_monomials (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  10 * a^3 * b^2 / (-5 * a^2 * b) = -2 * a * b :=
by sorry

end division_of_monomials_l3117_311727


namespace units_digit_of_product_l3117_311726

theorem units_digit_of_product (n : ℕ) : 
  (2^2101 * 5^2102 * 11^2103) % 10 = 0 :=
sorry

end units_digit_of_product_l3117_311726


namespace equal_sum_groups_l3117_311779

/-- A function that checks if a list of natural numbers can be divided into three groups with equal sums -/
def canDivideIntoThreeEqualGroups (list : List Nat) : Prop :=
  ∃ (g1 g2 g3 : List Nat), 
    g1 ++ g2 ++ g3 = list ∧ 
    g1.sum = g2.sum ∧ 
    g2.sum = g3.sum

/-- The list of natural numbers from 1 to n -/
def naturalNumbersUpTo (n : Nat) : List Nat :=
  List.range n |>.map (· + 1)

/-- The main theorem stating the condition for when the natural numbers up to n can be divided into three groups with equal sums -/
theorem equal_sum_groups (n : Nat) : 
  canDivideIntoThreeEqualGroups (naturalNumbersUpTo n) ↔ 
  (∃ k : Nat, (k ≥ 2 ∧ (n = 3 * k ∨ n = 3 * k - 1))) :=
sorry

end equal_sum_groups_l3117_311779


namespace simple_interest_problem_l3117_311732

/-- Proves that given a principal of 5000, if increasing the interest rate by 3%
    results in 300 more interest over the same time period, then the time period is 2 years. -/
theorem simple_interest_problem (R : ℚ) (T : ℚ) : 
  (5000 * (R + 3) / 100 * T = 5000 * R / 100 * T + 300) → T = 2 := by
  sorry

end simple_interest_problem_l3117_311732


namespace total_legs_of_daniels_animals_l3117_311760

/-- The number of legs an animal has -/
def legs (animal : String) : ℕ :=
  match animal with
  | "horse" => 4
  | "dog" => 4
  | "cat" => 4
  | "turtle" => 4
  | "goat" => 4
  | _ => 0

/-- Daniel's collection of animals -/
def daniels_animals : List (String × ℕ) :=
  [("horse", 2), ("dog", 5), ("cat", 7), ("turtle", 3), ("goat", 1)]

/-- Theorem: The total number of legs of Daniel's animals is 72 -/
theorem total_legs_of_daniels_animals :
  (daniels_animals.map (fun (animal, count) => count * legs animal)).sum = 72 := by
  sorry

end total_legs_of_daniels_animals_l3117_311760


namespace real_estate_pricing_l3117_311787

theorem real_estate_pricing (retail_price : ℝ) (retail_price_pos : retail_price > 0) :
  let z_price := retail_price * (1 - 0.3)
  let x_price := z_price * (1 - 0.15)
  let y_price := ((z_price + x_price) / 2) * (1 - 0.4)
  y_price / x_price = 0.653 := by
sorry

end real_estate_pricing_l3117_311787


namespace father_son_age_relationship_l3117_311737

/-- Represents the age relationship between a father and his son Ronit -/
structure AgeRelationship where
  ronit_age : ℕ
  father_age : ℕ
  years_passed : ℕ

/-- The conditions of the problem -/
def age_conditions (ar : AgeRelationship) : Prop :=
  (ar.father_age = 4 * ar.ronit_age) ∧
  (ar.father_age + ar.years_passed = (5/2) * (ar.ronit_age + ar.years_passed)) ∧
  (ar.father_age + ar.years_passed + 8 = 2 * (ar.ronit_age + ar.years_passed + 8))

theorem father_son_age_relationship :
  ∃ ar : AgeRelationship, age_conditions ar ∧ ar.years_passed = 8 := by
  sorry

end father_son_age_relationship_l3117_311737


namespace rice_distribution_l3117_311707

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 35 / 2 →
  num_containers = 4 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce) / num_containers = 70 := by
  sorry

end rice_distribution_l3117_311707


namespace tan_alpha_value_l3117_311716

theorem tan_alpha_value (α : Real) (h : Real.tan α = 3/4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64/25 := by
  sorry

end tan_alpha_value_l3117_311716


namespace sin_70_65_minus_sin_20_25_l3117_311793

theorem sin_70_65_minus_sin_20_25 : 
  Real.sin (70 * π / 180) * Real.sin (65 * π / 180) - 
  Real.sin (20 * π / 180) * Real.sin (25 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end sin_70_65_minus_sin_20_25_l3117_311793


namespace building_area_theorem_l3117_311744

/-- Represents a rectangular building with three floors -/
structure Building where
  breadth : ℝ
  length : ℝ
  area_per_floor : ℝ

/-- Calculates the total painting cost for the building -/
def total_painting_cost (b : Building) : ℝ :=
  b.area_per_floor * (3 + 4 + 5)

/-- Theorem: If the length is 200% more than the breadth and the total painting cost is 3160,
    then the total area of the building is 790 sq m -/
theorem building_area_theorem (b : Building) :
  b.length = 3 * b.breadth →
  total_painting_cost b = 3160 →
  3 * b.area_per_floor = 790 :=
by
  sorry

#check building_area_theorem

end building_area_theorem_l3117_311744


namespace cos_alpha_value_l3117_311763

def point_on_terminal_side (α : Real) (x y : Real) : Prop :=
  ∃ (r : Real), r > 0 ∧ x = r * Real.cos α ∧ y = r * Real.sin α

theorem cos_alpha_value (α : Real) :
  point_on_terminal_side α 1 3 → Real.cos α = 1 / Real.sqrt 10 := by
  sorry

end cos_alpha_value_l3117_311763


namespace factorial_difference_quotient_l3117_311743

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_difference_quotient : (factorial 13 - factorial 12) / factorial 10 = 1584 := by
  sorry

end factorial_difference_quotient_l3117_311743


namespace shopkeeper_profit_l3117_311780

theorem shopkeeper_profit (c s : ℝ) (p : ℝ) (h1 : c > 0) (h2 : s > c) :
  s = c * (1 + p / 100) ∧ 
  s = (0.9 * c) * (1 + (p + 12) / 100) →
  p = 8 := by
sorry

end shopkeeper_profit_l3117_311780


namespace inscribed_cylinder_height_l3117_311705

theorem inscribed_cylinder_height (r_cylinder : ℝ) (r_sphere : ℝ) :
  r_cylinder = 3 →
  r_sphere = 7 →
  let h := 2 * (2 * Real.sqrt 10)
  h = 2 * Real.sqrt (r_sphere^2 - r_cylinder^2) := by
  sorry

end inscribed_cylinder_height_l3117_311705


namespace square_sum_equals_eleven_halves_l3117_311745

theorem square_sum_equals_eleven_halves (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 4) : 
  a^2 + b^2 = 11/2 := by
sorry

end square_sum_equals_eleven_halves_l3117_311745


namespace rectangular_field_dimensions_l3117_311724

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 72 → m = (1 + Real.sqrt 1153) / 6 := by
sorry

end rectangular_field_dimensions_l3117_311724


namespace bamboo_nine_nodes_l3117_311746

theorem bamboo_nine_nodes (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 + a 2 + a 3 + a 4 = 3 →   -- sum of first 4 terms
  a 7 + a 8 + a 9 = 4 →         -- sum of last 3 terms
  a 1 + a 3 + a 9 = 17/6 :=     -- sum of 1st, 3rd, and 9th terms
by sorry

end bamboo_nine_nodes_l3117_311746


namespace total_seashells_l3117_311774

/-- The number of seashells found by Mary -/
def x : ℝ := 2

/-- The number of seashells found by Keith -/
def y : ℝ := 5

/-- The percentage of cracked seashells found by Mary -/
def m : ℝ := 0.5

/-- The percentage of cracked seashells found by Keith -/
def k : ℝ := 0.6

/-- The total number of seashells found by Mary and Keith -/
def T : ℝ := x + y

/-- The total number of cracked seashells -/
def z : ℝ := m * x + k * y

theorem total_seashells : T = 7 := by
  sorry

end total_seashells_l3117_311774


namespace remaining_pictures_l3117_311741

/-- The number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := 15

/-- The number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- The number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- The theorem states that the number of pictures Megan still has from her vacation is 2 -/
theorem remaining_pictures :
  zoo_pictures + museum_pictures - deleted_pictures = 2 := by sorry

end remaining_pictures_l3117_311741


namespace inscribed_square_area_l3117_311709

/-- The area of a regular square inscribed in a circle with area 324π is 648 square units. -/
theorem inscribed_square_area (circle_area : ℝ) (h : circle_area = 324 * Real.pi) :
  let r : ℝ := Real.sqrt (circle_area / Real.pi)
  let square_side : ℝ := Real.sqrt 2 * r
  square_side ^ 2 = 648 := by sorry

end inscribed_square_area_l3117_311709


namespace square_park_fencing_cost_l3117_311702

/-- The cost of fencing one side of a square park -/
def cost_per_side : ℕ := 43

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- The total cost of fencing a square park -/
def total_cost : ℕ := cost_per_side * num_sides

/-- Theorem: The total cost of fencing a square park is $172 -/
theorem square_park_fencing_cost :
  total_cost = 172 := by
  sorry

end square_park_fencing_cost_l3117_311702


namespace haley_halloween_candy_l3117_311789

/-- Represents the number of candy pieces Haley scored on Halloween -/
def initial_candy : ℕ := sorry

/-- Represents the number of candy pieces Haley ate -/
def eaten_candy : ℕ := 17

/-- Represents the number of candy pieces Haley received from her sister -/
def received_candy : ℕ := 19

/-- Represents the number of candy pieces Haley has now -/
def current_candy : ℕ := 35

/-- Proves that Haley scored 33 pieces of candy on Halloween -/
theorem haley_halloween_candy : initial_candy = 33 :=
  by
    have h : initial_candy - eaten_candy + received_candy = current_candy := sorry
    sorry

end haley_halloween_candy_l3117_311789


namespace camp_grouping_l3117_311757

theorem camp_grouping (total_children : ℕ) (max_group_size : ℕ) (h1 : total_children = 30) (h2 : max_group_size = 12) :
  ∃ (group_size : ℕ) (num_groups : ℕ),
    group_size ≤ max_group_size ∧
    group_size * num_groups = total_children ∧
    ∀ (k : ℕ), k ≤ max_group_size → k * (total_children / k) = total_children → num_groups ≤ (total_children / k) :=
by
  sorry

end camp_grouping_l3117_311757


namespace total_yen_is_correct_l3117_311749

/-- Represents the total assets of a family in various currencies and investments -/
structure FamilyAssets where
  bahamian_dollars : ℝ
  us_dollars : ℝ
  euros : ℝ
  checking_account1 : ℝ
  checking_account2 : ℝ
  savings_account1 : ℝ
  savings_account2 : ℝ
  stocks : ℝ
  bonds : ℝ
  mutual_funds : ℝ

/-- Exchange rates for different currencies to Japanese yen -/
structure ExchangeRates where
  bahamian_to_yen : ℝ
  usd_to_yen : ℝ
  euro_to_yen : ℝ

/-- Calculates the total amount of yen from all assets -/
def total_yen (assets : FamilyAssets) (rates : ExchangeRates) : ℝ :=
  assets.bahamian_dollars * rates.bahamian_to_yen +
  assets.us_dollars * rates.usd_to_yen +
  assets.euros * rates.euro_to_yen +
  assets.checking_account1 +
  assets.checking_account2 +
  assets.savings_account1 +
  assets.savings_account2 +
  assets.stocks +
  assets.bonds +
  assets.mutual_funds

/-- Theorem stating that the total amount of yen is 1,716,611 -/
theorem total_yen_is_correct (assets : FamilyAssets) (rates : ExchangeRates) :
  assets.bahamian_dollars = 5000 →
  assets.us_dollars = 2000 →
  assets.euros = 3000 →
  assets.checking_account1 = 15000 →
  assets.checking_account2 = 6359 →
  assets.savings_account1 = 5500 →
  assets.savings_account2 = 3102 →
  assets.stocks = 200000 →
  assets.bonds = 150000 →
  assets.mutual_funds = 120000 →
  rates.bahamian_to_yen = 122.13 →
  rates.usd_to_yen = 110.25 →
  rates.euro_to_yen = 128.50 →
  total_yen assets rates = 1716611 := by
  sorry

end total_yen_is_correct_l3117_311749


namespace work_comparison_l3117_311723

/-- Represents the amount of work that can be done by a group of people in a given number of days -/
structure WorkCapacity where
  people : ℕ
  days : ℕ
  work : ℝ

/-- The work capacity is directly proportional to the number of people and days -/
axiom work_proportional {w1 w2 : WorkCapacity} : 
  w1.work / w2.work = (w1.people * w1.days : ℝ) / (w2.people * w2.days)

theorem work_comparison (w1 w2 : WorkCapacity) 
  (h1 : w1.people = 3 ∧ w1.days = 3)
  (h2 : w2.people = 8 ∧ w2.days = 3)
  (h3 : w2.work = 8 * w1.work) :
  w1.work = 3 * w1.work := by
  sorry

end work_comparison_l3117_311723


namespace teachers_in_middle_probability_l3117_311710

def num_students : ℕ := 3
def num_teachers : ℕ := 2
def num_parents : ℕ := 3
def total_people : ℕ := num_students + num_teachers + num_parents

def probability_teachers_in_middle : ℚ :=
  (Nat.factorial (total_people - num_teachers)) / (Nat.factorial total_people)

theorem teachers_in_middle_probability :
  probability_teachers_in_middle = 1 / 56 := by
  sorry

end teachers_in_middle_probability_l3117_311710


namespace joan_seashells_problem_l3117_311765

theorem joan_seashells_problem (initial : ℕ) (remaining : ℕ) (sam_to_lily_ratio : ℕ) :
  initial = 70 →
  remaining = 27 →
  sam_to_lily_ratio = 2 →
  ∃ (sam lily : ℕ),
    initial = remaining + sam + lily ∧
    sam = sam_to_lily_ratio * lily ∧
    sam = 28 :=
by sorry

end joan_seashells_problem_l3117_311765


namespace milburg_population_l3117_311770

theorem milburg_population :
  let grown_ups : ℕ := 5256
  let children : ℕ := 2987
  grown_ups + children = 8243 := by
  sorry

end milburg_population_l3117_311770


namespace village_population_theorem_l3117_311791

theorem village_population_theorem (total_population : ℕ) 
  (h1 : total_population = 800) 
  (h2 : total_population % 4 = 0) 
  (h3 : 3 * (total_population / 4) = total_population - (total_population / 4)) :
  total_population / 4 = 200 :=
sorry

end village_population_theorem_l3117_311791


namespace f_composition_equals_constant_l3117_311750

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then 2 * z ^ 2 else -3 * z ^ 2

-- State the theorem
theorem f_composition_equals_constant : f (f (f (f (1 + I)))) = (-28311552 : ℂ) := by
  sorry

end f_composition_equals_constant_l3117_311750


namespace loads_required_l3117_311755

def washing_machine_capacity : ℕ := 9
def total_clothing : ℕ := 27

theorem loads_required : (total_clothing + washing_machine_capacity - 1) / washing_machine_capacity = 3 := by
  sorry

end loads_required_l3117_311755


namespace pen_diary_cost_l3117_311762

/-- Given that 6 pens and 5 diaries cost $6.10, and 3 pens and 4 diaries cost $4.60,
    prove that 12 pens and 8 diaries cost $10.16 -/
theorem pen_diary_cost : ∃ (pen_cost diary_cost : ℝ),
  (6 * pen_cost + 5 * diary_cost = 6.10) ∧
  (3 * pen_cost + 4 * diary_cost = 4.60) ∧
  (12 * pen_cost + 8 * diary_cost = 10.16) := by
  sorry


end pen_diary_cost_l3117_311762


namespace initial_men_is_100_l3117_311706

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℝ
  completedLength : ℝ
  completedDays : ℝ
  extraMen : ℕ

/-- Calculates the initial number of men employed in the road project -/
def initialMenEmployed (project : RoadProject) : ℕ :=
  sorry

/-- Theorem stating that the initial number of men employed is 100 -/
theorem initial_men_is_100 (project : RoadProject) 
  (h1 : project.totalLength = 15)
  (h2 : project.totalDays = 300)
  (h3 : project.completedLength = 2.5)
  (h4 : project.completedDays = 100)
  (h5 : project.extraMen = 60) :
  initialMenEmployed project = 100 := by
  sorry

#check initial_men_is_100

end initial_men_is_100_l3117_311706


namespace quadratic_term_elimination_l3117_311776

theorem quadratic_term_elimination (m : ℝ) : 
  (∀ x : ℝ, 36 * x^2 - 3 * x + 5 - (-3 * x^3 - 12 * m * x^2 + 5 * x - 7) = 3 * x^3 - 8 * x + 12) → 
  m^3 = -27 := by
sorry

end quadratic_term_elimination_l3117_311776


namespace unique_positive_solution_l3117_311786

theorem unique_positive_solution (x y z : ℝ) : 
  x > 0 →
  x * y + 3 * x + 2 * y = 12 →
  y * z + 5 * y + 3 * z = 18 →
  x * z + 2 * x + 3 * z = 18 →
  x = 4 := by
sorry

end unique_positive_solution_l3117_311786


namespace greatest_three_digit_number_multiple_condition_l3117_311700

theorem greatest_three_digit_number_multiple_condition : ∃ n : ℕ,
  (n ≤ 999) ∧ 
  (n ≥ 100) ∧
  (∃ k : ℕ, n = 9 * k + 2) ∧
  (∃ m : ℕ, n = 7 * m + 4) ∧
  (∀ x : ℕ, x ≤ 999 ∧ x ≥ 100 ∧ (∃ k : ℕ, x = 9 * k + 2) ∧ (∃ m : ℕ, x = 7 * m + 4) → x ≤ n) ∧
  n = 956 :=
by
  sorry

end greatest_three_digit_number_multiple_condition_l3117_311700


namespace hexagon_segment_probability_l3117_311759

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of short diagonals in a regular hexagon -/
def num_short_diagonals : ℕ := 3

/-- The number of long diagonals in a regular hexagon -/
def num_long_diagonals : ℕ := 6

/-- The probability of selecting two segments of the same length from T -/
def prob_same_length : ℚ := 11/35

theorem hexagon_segment_probability :
  let total := T.card
  let same_length_pairs := (num_sides.choose 2) + (num_short_diagonals.choose 2) + (num_long_diagonals.choose 2)
  prob_same_length = same_length_pairs / (total.choose 2) := by
  sorry

end hexagon_segment_probability_l3117_311759


namespace cubic_difference_zero_l3117_311725

theorem cubic_difference_zero (a b : ℝ) (h1 : a - b = 1) (h2 : a * b ≠ 0) :
  a^3 - b^3 - a*b - a^2 - b^2 = 0 := by
  sorry

end cubic_difference_zero_l3117_311725


namespace janet_earnings_per_hour_l3117_311730

-- Define the payment rates for each type of post
def text_post_rate : ℚ := 0.25
def image_post_rate : ℚ := 0.30
def video_post_rate : ℚ := 0.40

-- Define the number of posts checked in an hour
def text_posts_per_hour : ℕ := 130
def image_posts_per_hour : ℕ := 90
def video_posts_per_hour : ℕ := 30

-- Define the USD to EUR exchange rate
def usd_to_eur_rate : ℚ := 0.85

-- Calculate the earnings per hour in EUR
def earnings_per_hour_eur : ℚ :=
  (text_post_rate * text_posts_per_hour +
   image_post_rate * image_posts_per_hour +
   video_post_rate * video_posts_per_hour) * usd_to_eur_rate

-- Theorem to prove
theorem janet_earnings_per_hour :
  earnings_per_hour_eur = 60.775 := by sorry

end janet_earnings_per_hour_l3117_311730


namespace point_in_third_quadrant_l3117_311731

theorem point_in_third_quadrant (m : ℝ) : 
  let P : ℝ × ℝ := (-m^2 - 1, -1)
  P.1 < 0 ∧ P.2 < 0 :=
by sorry

end point_in_third_quadrant_l3117_311731


namespace circle_area_theorem_l3117_311752

-- Define the center and point on the circle
def center : ℝ × ℝ := (-2, 5)
def point : ℝ × ℝ := (8, -4)

-- Calculate the squared distance between two points
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2

-- Define the theorem
theorem circle_area_theorem :
  let r := Real.sqrt (distance_squared center point)
  π * r^2 = 181 * π := by sorry

end circle_area_theorem_l3117_311752


namespace wine_bottle_cost_l3117_311711

/-- The cost of a bottle of wine with a cork, given the price of the cork and the price difference between a bottle without a cork and the cork itself. -/
theorem wine_bottle_cost (cork_price : ℝ) (price_difference : ℝ) : 
  cork_price = 0.05 →
  price_difference = 2.00 →
  cork_price + (cork_price + price_difference) = 2.10 :=
by sorry

end wine_bottle_cost_l3117_311711


namespace chocolate_bars_distribution_l3117_311795

theorem chocolate_bars_distribution (total_bars : ℕ) (num_people : ℕ) 
  (h1 : total_bars = 12) (h2 : num_people = 3) :
  2 * (total_bars / num_people) = 8 := by
  sorry

end chocolate_bars_distribution_l3117_311795


namespace a_share_of_profit_l3117_311729

/-- Calculate A's share of the profit in a partnership business -/
theorem a_share_of_profit (a_investment b_investment c_investment total_profit : ℕ) :
  a_investment = 6300 →
  b_investment = 4200 →
  c_investment = 10500 →
  total_profit = 12700 →
  (a_investment * total_profit) / (a_investment + b_investment + c_investment) = 3810 :=
by sorry

end a_share_of_profit_l3117_311729


namespace painting_areas_l3117_311721

/-- Represents the areas painted in square decimeters -/
structure PaintedAreas where
  blue : ℝ
  green : ℝ
  yellow : ℝ

/-- The total amount of each paint color available in square decimeters -/
def total_paint : ℝ := 38

/-- Theorem stating the correct areas given the painting conditions -/
theorem painting_areas : ∃ (areas : PaintedAreas),
  -- All paint is used
  areas.blue + areas.yellow + areas.green = 2 * total_paint ∧
  -- Green paint mixture ratio
  areas.green = (2 * areas.yellow + areas.blue) / 3 ∧
  -- Grass area is 6 more than sky area
  areas.green = areas.blue + 6 ∧
  -- Correct areas
  areas.blue = 27 ∧
  areas.green = 33 ∧
  areas.yellow = 16 := by
  sorry

end painting_areas_l3117_311721


namespace prob_at_least_one_from_three_suits_l3117_311790

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Number of cards in each suit -/
def cardsPerSuit : ℕ := 13

/-- Number of cards drawn -/
def numDraws : ℕ := 5

/-- Number of specific suits considered -/
def numSpecificSuits : ℕ := 3

/-- Probability of drawing a card from the specific suits in one draw -/
def probSpecificSuits : ℚ := (cardsPerSuit * numSpecificSuits) / standardDeck

/-- Probability of drawing a card not from the specific suits in one draw -/
def probNotSpecificSuits : ℚ := 1 - probSpecificSuits

/-- 
Theorem: The probability of drawing at least one card from each of three specific suits 
when choosing five cards with replacement from a standard 52-card deck is 1023/1024.
-/
theorem prob_at_least_one_from_three_suits : 
  1 - probNotSpecificSuits ^ numDraws = 1023 / 1024 := by sorry

end prob_at_least_one_from_three_suits_l3117_311790


namespace max_correct_percentage_l3117_311754

theorem max_correct_percentage
  (total : ℝ)
  (solo_portion : ℝ)
  (together_portion : ℝ)
  (chloe_solo_correct : ℝ)
  (chloe_overall_correct : ℝ)
  (max_solo_correct : ℝ)
  (h1 : solo_portion = 2/3)
  (h2 : together_portion = 1/3)
  (h3 : solo_portion + together_portion = 1)
  (h4 : chloe_solo_correct = 0.7)
  (h5 : chloe_overall_correct = 0.82)
  (h6 : max_solo_correct = 0.85)
  : max_solo_correct * solo_portion + (chloe_overall_correct - chloe_solo_correct * solo_portion) = 0.92 := by
  sorry

end max_correct_percentage_l3117_311754


namespace lottery_probability_l3117_311756

def powerball_count : ℕ := 30
def luckyball_count : ℕ := 50
def luckyball_draw : ℕ := 6

theorem lottery_probability :
  (1 : ℚ) / (powerball_count * (Nat.choose luckyball_count luckyball_draw)) = 1 / 476721000 :=
by sorry

end lottery_probability_l3117_311756


namespace jeff_running_schedule_l3117_311792

/-- Jeff's running schedule problem -/
theorem jeff_running_schedule 
  (weekday_run : ℕ) -- Planned running time per weekday in minutes
  (thursday_cut : ℕ) -- Minutes cut from Thursday's run
  (total_time : ℕ) -- Total running time for the week in minutes
  (h1 : weekday_run = 60)
  (h2 : thursday_cut = 20)
  (h3 : total_time = 290) :
  total_time - (4 * weekday_run + (weekday_run - thursday_cut)) = 10 :=
by sorry

end jeff_running_schedule_l3117_311792


namespace number_divided_by_quarter_l3117_311783

theorem number_divided_by_quarter : ∀ x : ℝ, x / 0.25 = 400 → x = 100 := by
  sorry

end number_divided_by_quarter_l3117_311783


namespace sin_40_minus_sin_80_l3117_311736

theorem sin_40_minus_sin_80 : 
  Real.sin (40 * π / 180) - Real.sin (80 * π / 180) = 
    Real.sin (40 * π / 180) * (1 - 2 * Real.sqrt (1 - Real.sin (40 * π / 180) ^ 2)) := by
  sorry

end sin_40_minus_sin_80_l3117_311736


namespace fraction_to_decimal_equiv_l3117_311772

theorem fraction_to_decimal_equiv : (5 : ℚ) / 8 = 0.625 := by sorry

end fraction_to_decimal_equiv_l3117_311772


namespace expression_equals_three_l3117_311718

-- Define the expression
def expression : ℚ := -25 + 7 * ((8 / 4) ^ 2)

-- Theorem statement
theorem expression_equals_three : expression = 3 := by
  sorry

end expression_equals_three_l3117_311718


namespace perpendicular_diagonals_not_sufficient_for_rhombus_l3117_311735

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define diagonals of a quadrilateral
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((q.A.1 - q.C.1, q.A.2 - q.C.2), (q.B.1 - q.D.1, q.B.2 - q.D.2))

-- Define perpendicularity of two vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  let (AC, BD) := diagonals q
  perpendicular AC BD ∧ 
  AC.1^2 + AC.2^2 = BD.1^2 + BD.2^2 ∧
  (AC.1 / 2, AC.2 / 2) = (BD.1 / 2, BD.2 / 2)

-- Statement to prove
theorem perpendicular_diagonals_not_sufficient_for_rhombus :
  ∃ (q : Quadrilateral), 
    (let (AC, BD) := diagonals q; perpendicular AC BD) ∧ 
    ¬is_rhombus q :=
sorry

end perpendicular_diagonals_not_sufficient_for_rhombus_l3117_311735


namespace x_values_l3117_311753

theorem x_values (x : ℝ) : (|2000 * x + 2000| = 20 * 2000) → (x = 19 ∨ x = -21) := by
  sorry

end x_values_l3117_311753


namespace blood_cell_count_l3117_311766

theorem blood_cell_count (total : ℕ) (first_sample : ℕ) (second_sample : ℕ) 
  (h1 : total = 7341)
  (h2 : first_sample = 4221)
  (h3 : total = first_sample + second_sample) : 
  second_sample = 3120 := by
  sorry

end blood_cell_count_l3117_311766


namespace sally_peaches_theorem_l3117_311739

/-- Represents the number of peaches Sally picked at the orchard -/
def peaches_picked (initial total : ℕ) : ℕ := total - initial

/-- Theorem stating that the number of peaches Sally picked is the difference between her total and initial peaches -/
theorem sally_peaches_theorem (initial total : ℕ) (h : initial ≤ total) :
  peaches_picked initial total = total - initial :=
by sorry

end sally_peaches_theorem_l3117_311739


namespace intersection_point_sum_l3117_311748

theorem intersection_point_sum (c d : ℝ) :
  (∃ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d) →
  (3 = (1/3) * 6 + c ∧ 6 = (1/3) * 3 + d) →
  c + d = 6 := by
sorry

end intersection_point_sum_l3117_311748


namespace cost_of_stationery_l3117_311703

/-- Given the cost of erasers, pens, and markers satisfying certain conditions,
    prove that the total cost of 3 erasers, 4 pens, and 6 markers is 520 rubles. -/
theorem cost_of_stationery (E P M : ℕ) 
    (h1 : E + 3 * P + 2 * M = 240)
    (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := by
  sorry


end cost_of_stationery_l3117_311703


namespace eighth_term_of_specific_arithmetic_sequence_l3117_311722

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 8th term of the arithmetic sequence with first term 1 and common difference 3 is 22 -/
theorem eighth_term_of_specific_arithmetic_sequence :
  arithmeticSequenceTerm 1 3 8 = 22 := by
  sorry

end eighth_term_of_specific_arithmetic_sequence_l3117_311722


namespace function_always_positive_l3117_311778

theorem function_always_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, 2 * f x + x * deriv f x > 0) : 
  ∀ x, f x > 0 := by sorry

end function_always_positive_l3117_311778


namespace tan_equation_solution_l3117_311796

theorem tan_equation_solution (θ : Real) :
  2 * Real.tan θ - Real.tan (θ + π/4) = 7 → Real.tan θ = 2 := by
  sorry

end tan_equation_solution_l3117_311796


namespace college_choices_theorem_l3117_311715

/-- The number of colleges --/
def n : ℕ := 6

/-- The number of colleges to be chosen --/
def k : ℕ := 3

/-- The number of colleges with scheduling conflict --/
def conflict : ℕ := 2

/-- Function to calculate the number of ways to choose colleges --/
def chooseColleges (n k conflict : ℕ) : ℕ :=
  Nat.choose (n - conflict) k + conflict * Nat.choose (n - conflict) (k - 1)

/-- Theorem stating that the number of ways to choose colleges is 16 --/
theorem college_choices_theorem :
  chooseColleges n k conflict = 16 := by sorry

end college_choices_theorem_l3117_311715


namespace sequence_exists_l3117_311799

def is_valid_sequence (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, seq n + seq (n + 1) + seq (n + 2) = 15

def is_repeating (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, seq n = seq (n + 3)

theorem sequence_exists : ∃ seq : ℕ → ℕ, is_valid_sequence seq ∧ is_repeating seq :=
sorry

end sequence_exists_l3117_311799


namespace complex_product_example_l3117_311758

theorem complex_product_example : 
  let z₁ : ℂ := -1 + 2 * Complex.I
  let z₂ : ℂ := 2 + Complex.I
  z₁ * z₂ = -4 + 3 * Complex.I := by
sorry

end complex_product_example_l3117_311758


namespace school_population_after_new_students_l3117_311740

theorem school_population_after_new_students (initial_avg_age : ℝ) (new_students : ℕ) 
  (new_students_avg_age : ℝ) (avg_age_decrease : ℝ) :
  initial_avg_age = 48 →
  new_students = 120 →
  new_students_avg_age = 32 →
  avg_age_decrease = 4 →
  ∃ (initial_students : ℕ),
    (initial_students + new_students : ℝ) * (initial_avg_age - avg_age_decrease) = 
    initial_students * initial_avg_age + new_students * new_students_avg_age ∧
    initial_students + new_students = 480 :=
by sorry

end school_population_after_new_students_l3117_311740


namespace contest_scores_l3117_311738

theorem contest_scores (x y : ℝ) : 
  (9 + 8.7 + 9.3 + x + y) / 5 = 9 →
  ((9 - 9)^2 + (8.7 - 9)^2 + (9.3 - 9)^2 + (x - 9)^2 + (y - 9)^2) / 5 = 0.1 →
  |x - y| = 0.8 := by
sorry

end contest_scores_l3117_311738


namespace tree_leaves_theorem_l3117_311719

/-- Calculates the number of leaves remaining on a tree after three weeks of shedding --/
def leaves_remaining (initial_leaves : ℕ) : ℕ :=
  let first_week_remaining := initial_leaves - (2 * initial_leaves / 5)
  let second_week_shed := (40 * first_week_remaining) / 100
  let second_week_remaining := first_week_remaining - second_week_shed
  let third_week_shed := (3 * second_week_shed) / 4
  second_week_remaining - third_week_shed

/-- Theorem stating that a tree with 1000 initial leaves will have 180 leaves remaining after three weeks of shedding --/
theorem tree_leaves_theorem : leaves_remaining 1000 = 180 := by
  sorry

end tree_leaves_theorem_l3117_311719


namespace sticker_ratio_l3117_311785

/-- Proves that the ratio of silver stickers to gold stickers is 2:1 --/
theorem sticker_ratio :
  ∀ (gold silver bronze : ℕ),
  gold = 50 →
  bronze = silver - 20 →
  gold + silver + bronze = 5 * 46 →
  silver / gold = 2 := by
  sorry

end sticker_ratio_l3117_311785


namespace gcd_of_1054_and_986_l3117_311708

theorem gcd_of_1054_and_986 : Nat.gcd 1054 986 = 34 := by
  sorry

end gcd_of_1054_and_986_l3117_311708


namespace overall_gain_loss_percent_zero_l3117_311712

def article_A_cost : ℝ := 600
def article_B_cost : ℝ := 700
def article_C_cost : ℝ := 800
def article_A_sell : ℝ := 450
def article_B_sell : ℝ := 750
def article_C_sell : ℝ := 900

def total_cost : ℝ := article_A_cost + article_B_cost + article_C_cost
def total_sell : ℝ := article_A_sell + article_B_sell + article_C_sell

theorem overall_gain_loss_percent_zero :
  (total_sell - total_cost) / total_cost * 100 = 0 := by sorry

end overall_gain_loss_percent_zero_l3117_311712


namespace age_of_b_l3117_311714

/-- Given three people a, b, and c, prove that if their average age is 25 years
    and the average age of a and c is 29 years, then the age of b is 17 years. -/
theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 25 → (a + c) / 2 = 29 → b = 17 := by
  sorry

end age_of_b_l3117_311714


namespace circle_contains_at_least_250_points_l3117_311782

/-- A circle on a grid --/
structure GridCircle where
  radius : ℝ
  gridSize : ℝ

/-- The number of grid points inside a circle --/
def gridPointsInside (c : GridCircle) : ℕ :=
  sorry

/-- Theorem: A circle with radius 10 on a unit grid contains at least 250 grid points --/
theorem circle_contains_at_least_250_points (c : GridCircle) 
  (h1 : c.radius = 10)
  (h2 : c.gridSize = 1) : 
  gridPointsInside c ≥ 250 := by
  sorry

end circle_contains_at_least_250_points_l3117_311782


namespace tangent_line_equation_l3117_311701

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

/-- The point of tangency -/
def P : ℝ × ℝ := (-1, 3)

/-- The slope of the tangent line at point P -/
def m : ℝ := f' P.1

/-- The equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := m * (x - P.1) + P.2

theorem tangent_line_equation :
  ∀ x : ℝ, tangent_line x = -4 * x - 1 :=
by sorry

end tangent_line_equation_l3117_311701


namespace committee_with_chair_count_l3117_311777

theorem committee_with_chair_count : 
  let total_students : ℕ := 8
  let committee_size : ℕ := 5
  let committee_count : ℕ := Nat.choose total_students committee_size
  let chair_choices : ℕ := committee_size
  committee_count * chair_choices = 280 := by
sorry

end committee_with_chair_count_l3117_311777


namespace min_value_theorem_l3117_311720

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + a*b + a*c + b*c = 6 + 2 * Real.sqrt 5) :
  3*a + b + 2*c ≥ 2 * Real.sqrt 10 + 2 * Real.sqrt 2 :=
by sorry

end min_value_theorem_l3117_311720


namespace john_paid_21_dollars_l3117_311767

/-- Calculates the amount John paid for candy bars -/
def john_payment (total_bars : ℕ) (dave_bars : ℕ) (cost_per_bar : ℚ) : ℚ :=
  (total_bars - dave_bars) * cost_per_bar

/-- Proves that John paid $21 for the candy bars -/
theorem john_paid_21_dollars (total_bars : ℕ) (dave_bars : ℕ) (cost_per_bar : ℚ)
  (h1 : total_bars = 20)
  (h2 : dave_bars = 6)
  (h3 : cost_per_bar = 3/2) :
  john_payment total_bars dave_bars cost_per_bar = 21 := by
  sorry

end john_paid_21_dollars_l3117_311767


namespace natasha_quarters_l3117_311751

theorem natasha_quarters : ∃ n : ℕ,
  8 < n ∧ n < 80 ∧
  n % 4 = 3 ∧
  n % 5 = 1 ∧
  n % 7 = 3 ∧
  n = 31 := by
sorry

end natasha_quarters_l3117_311751


namespace tan_2alpha_proof_l3117_311775

theorem tan_2alpha_proof (α : Real) (h : Real.sin α + 2 * Real.cos α = Real.sqrt 10 / 2) :
  Real.tan (2 * α) = -3/4 := by
  sorry

end tan_2alpha_proof_l3117_311775
