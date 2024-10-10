import Mathlib

namespace unripe_oranges_count_l3166_316667

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges : ℕ := 44

/-- The difference between the number of sacks of ripe and unripe oranges harvested per day -/
def difference : ℕ := 19

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges : ℕ := ripe_oranges - difference

theorem unripe_oranges_count : unripe_oranges = 25 := by
  sorry

end unripe_oranges_count_l3166_316667


namespace chalk_boxes_l3166_316645

theorem chalk_boxes (total_chalk : ℕ) (chalk_per_box : ℕ) (h1 : total_chalk = 3484) (h2 : chalk_per_box = 18) :
  (total_chalk + chalk_per_box - 1) / chalk_per_box = 194 := by
  sorry

end chalk_boxes_l3166_316645


namespace program_production_cost_l3166_316694

/-- The cost to produce a program for a college football game. -/
def cost_to_produce : ℝ :=
  sorry

/-- Theorem: Given the conditions, the cost to produce a program is 5500 rupees. -/
theorem program_production_cost :
  let advertisement_revenue : ℝ := 15000
  let copies_sold : ℝ := 35000
  let price_per_copy : ℝ := 0.50
  let desired_profit : ℝ := 8000
  cost_to_produce = advertisement_revenue + (copies_sold * price_per_copy) - (advertisement_revenue + desired_profit) :=
by
  sorry

end program_production_cost_l3166_316694


namespace yellow_green_weight_difference_l3166_316673

/-- The weight difference between two blocks -/
def weight_difference (yellow_weight green_weight : Real) : Real :=
  yellow_weight - green_weight

/-- Theorem stating the weight difference between yellow and green blocks -/
theorem yellow_green_weight_difference :
  let yellow_weight : Real := 0.6
  let green_weight : Real := 0.4
  weight_difference yellow_weight green_weight = 0.2 := by
  sorry

end yellow_green_weight_difference_l3166_316673


namespace percentage_problem_l3166_316636

theorem percentage_problem (x : ℝ) : 
  (x / 100) * 25 + 5.4 = 9.15 → x = 15 := by
  sorry

end percentage_problem_l3166_316636


namespace diameter_endpoint_theorem_l3166_316618

/-- A circle in a 2D coordinate plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A diameter of a circle --/
structure Diameter where
  circle : Circle
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- The theorem stating the relationship between the center and endpoints of a diameter --/
theorem diameter_endpoint_theorem (c : Circle) (d : Diameter) :
  c.center = (5, 2) ∧ d.circle = c ∧ d.endpoint1 = (0, -3) →
  d.endpoint2 = (10, 7) := by
  sorry

end diameter_endpoint_theorem_l3166_316618


namespace tim_soda_cans_l3166_316685

/-- The number of soda cans Tim has at the end of the scenario -/
def final_cans (initial : ℕ) (taken : ℕ) : ℕ :=
  let remaining := initial - taken
  remaining + (remaining / 2)

/-- Theorem stating that Tim ends up with 24 cans -/
theorem tim_soda_cans : final_cans 22 6 = 24 := by
  sorry

end tim_soda_cans_l3166_316685


namespace abs_eq_neg_iff_nonpositive_l3166_316679

theorem abs_eq_neg_iff_nonpositive (a : ℝ) : |a| = -a ↔ a ≤ 0 := by sorry

end abs_eq_neg_iff_nonpositive_l3166_316679


namespace log_2_base_10_bounds_l3166_316648

theorem log_2_base_10_bounds :
  (2^9 = 512) →
  (2^14 = 16384) →
  (10^3 = 1000) →
  (10^4 = 10000) →
  (2/7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (1/3 : ℝ) :=
by sorry

end log_2_base_10_bounds_l3166_316648


namespace sons_age_is_24_l3166_316661

/-- Proves that the son's age is 24 given the conditions of the problem -/
theorem sons_age_is_24 (son_age father_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end sons_age_is_24_l3166_316661


namespace waist_size_conversion_l3166_316626

/-- Converts inches to centimeters given the conversion rates and waist size --/
def inches_to_cm (inches_per_foot : ℚ) (cm_per_foot : ℚ) (waist_inches : ℚ) : ℚ :=
  (waist_inches / inches_per_foot) * cm_per_foot

/-- Theorem: Given the conversion rates and waist size, proves that 40 inches equals 100 cm --/
theorem waist_size_conversion :
  let inches_per_foot : ℚ := 10
  let cm_per_foot : ℚ := 25
  let waist_inches : ℚ := 40
  inches_to_cm inches_per_foot cm_per_foot waist_inches = 100 := by
  sorry

end waist_size_conversion_l3166_316626


namespace people_per_tent_l3166_316697

theorem people_per_tent 
  (total_people : ℕ) 
  (house_capacity : ℕ) 
  (num_tents : ℕ) 
  (h1 : total_people = 14) 
  (h2 : house_capacity = 4) 
  (h3 : num_tents = 5) :
  (total_people - house_capacity) / num_tents = 2 :=
by sorry

end people_per_tent_l3166_316697


namespace parabola_point_ordering_l3166_316601

/-- Given a parabola y = ax² + bx + c with 0 < 2a < b, and points A(1/2, y₁), B(0, y₂), C(-1, y₃) on the parabola,
    prove that y₁ > y₂ > y₃ -/
theorem parabola_point_ordering (a b c y₁ y₂ y₃ : ℝ) :
  0 < 2 * a → 2 * a < b →
  y₁ = a * (1/2)^2 + b * (1/2) + c →
  y₂ = c →
  y₃ = a * (-1)^2 + b * (-1) + c →
  y₁ > y₂ ∧ y₂ > y₃ := by
sorry

end parabola_point_ordering_l3166_316601


namespace m_range_correct_l3166_316609

/-- Statement p: For all x in ℝ, x^2 + mx + m/2 + 2 ≥ 0 always holds true -/
def statement_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + m/2 + 2 ≥ 0

/-- Statement q: The distance from the focus of the parabola y^2 = 2mx (where m > 0) to its directrix is greater than 1 -/
def statement_q (m : ℝ) : Prop :=
  m > 0 ∧ m/2 > 1

/-- The range of m that satisfies all conditions -/
def m_range : Set ℝ :=
  {m : ℝ | m > 4}

theorem m_range_correct :
  ∀ m : ℝ, (statement_p m ∨ statement_q m) ∧ ¬(statement_p m ∧ statement_q m) ↔ m ∈ m_range := by
  sorry

end m_range_correct_l3166_316609


namespace factor_implies_m_value_l3166_316625

theorem factor_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x + 6) ∣ (x^2 - m*x - 42)) → m = 1 := by
  sorry

end factor_implies_m_value_l3166_316625


namespace central_angle_nairobi_lima_l3166_316657

/-- Represents a point on Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on Earth -/
def centralAngle (p1 p2 : EarthPoint) : Real :=
  |p1.longitude - p2.longitude|

theorem central_angle_nairobi_lima :
  let nairobi : EarthPoint := { latitude := -1, longitude := 36 }
  let lima : EarthPoint := { latitude := -12, longitude := -77 }
  centralAngle nairobi lima = 113 := by
  sorry

end central_angle_nairobi_lima_l3166_316657


namespace problem_solution_l3166_316655

theorem problem_solution (a : ℝ) : 3 ∈ ({1, -a^2, a-1} : Set ℝ) → a = 4 := by
  sorry

end problem_solution_l3166_316655


namespace initial_oranges_l3166_316631

theorem initial_oranges (total : ℕ) 
  (h1 : total % 2 = 0)  -- Half of the oranges were ripe
  (h2 : (total / 2) % 4 = 0)  -- 1/4 of the ripe oranges were eaten
  (h3 : (total / 2) % 8 = 0)  -- 1/8 of the unripe oranges were eaten
  (h4 : total * 13 / 16 = 78)  -- 78 oranges were left uneaten in total
  : total = 96 := by
  sorry

end initial_oranges_l3166_316631


namespace union_of_A_and_B_l3166_316669

def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 < x ∧ x < 6} := by
  sorry

end union_of_A_and_B_l3166_316669


namespace f_property_P_implies_m_range_l3166_316691

/-- Property P(a) for a function f on domain D -/
def property_P (f : ℝ → ℝ) (D : Set ℝ) (a : ℝ) : Prop :=
  ∀ x₁ ∈ D, ∃ x₂ ∈ D, (x₁ + f x₂) / 2 = a

/-- The function f(x) = -x² + mx - 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - 3

/-- The domain of f(x) -/
def D : Set ℝ := {x : ℝ | x > 0}

theorem f_property_P_implies_m_range :
  ∀ m : ℝ, property_P (f m) D (1/2) → m ∈ {m : ℝ | m ≥ 4} := by sorry

end f_property_P_implies_m_range_l3166_316691


namespace tan_ratio_given_sin_relation_l3166_316638

theorem tan_ratio_given_sin_relation (α : Real) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * (Real.pi / 180))) :
  Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2 := by
  sorry

end tan_ratio_given_sin_relation_l3166_316638


namespace matthews_cracker_distribution_l3166_316650

theorem matthews_cracker_distribution (total_crackers : ℕ) (crackers_per_friend : ℕ) (num_friends : ℕ) :
  total_crackers = 36 →
  crackers_per_friend = 6 →
  total_crackers = crackers_per_friend * num_friends →
  num_friends = 6 := by
sorry

end matthews_cracker_distribution_l3166_316650


namespace mindmaster_codes_l3166_316693

/-- The number of colors available for the Mindmaster game -/
def num_colors : ℕ := 5

/-- The number of slots in the Mindmaster game -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes in the Mindmaster game -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes is 3125 -/
theorem mindmaster_codes : total_codes = 3125 := by
  sorry

end mindmaster_codes_l3166_316693


namespace chinese_table_tennis_team_arrangements_l3166_316619

/-- The number of players in the Chinese men's table tennis team -/
def total_players : ℕ := 6

/-- The number of players required for the team event -/
def team_size : ℕ := 3

/-- Calculates the number of permutations of k elements from n elements -/
def permutations (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

/-- The main theorem -/
theorem chinese_table_tennis_team_arrangements :
  permutations total_players team_size - permutations (total_players - 1) (team_size - 1) = 100 := by
  sorry


end chinese_table_tennis_team_arrangements_l3166_316619


namespace ellipse_foci_distance_l3166_316612

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  25 * x^2 + 100 * x + 9 * y^2 - 36 * y = 225

/-- The distance between the foci of the ellipse -/
def foci_distance : ℝ := 10.134

/-- Theorem: The distance between the foci of the ellipse defined by the given equation
    is approximately 10.134 -/
theorem ellipse_foci_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧
  (∀ x y : ℝ, ellipse_equation x y → abs (foci_distance - 10.134) < ε) :=
sorry

end ellipse_foci_distance_l3166_316612


namespace max_sum_given_constraints_l3166_316603

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end max_sum_given_constraints_l3166_316603


namespace algebraic_equality_l3166_316658

theorem algebraic_equality (a b c : ℝ) : a + b * c = (a + b) * (a + c) ↔ a + b + c = 1 := by
  sorry

end algebraic_equality_l3166_316658


namespace smallest_number_divisibility_l3166_316615

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisibility (n : ℕ) : 
  (is_divisible_by (n + 3) 12 ∧ 
   is_divisible_by (n + 3) 15 ∧ 
   is_divisible_by (n + 3) 40) →
  (∀ k : ℕ, k < n → 
    ¬(is_divisible_by (k + 3) 12 ∧ 
      is_divisible_by (k + 3) 15 ∧ 
      is_divisible_by (k + 3) 40)) →
  (∃ m : ℕ, m ≠ 12 ∧ m ≠ 15 ∧ m ≠ 40 ∧ 
    is_divisible_by (n + 3) m) →
  ∃ m : ℕ, m ≠ 12 ∧ m ≠ 15 ∧ m ≠ 40 ∧ 
    is_divisible_by (n + 3) m ∧ m = 2 :=
by sorry

end smallest_number_divisibility_l3166_316615


namespace eight_fifteen_div_sixtyfour_six_l3166_316628

theorem eight_fifteen_div_sixtyfour_six : 8^15 / 64^6 = 512 := by
  sorry

end eight_fifteen_div_sixtyfour_six_l3166_316628


namespace partial_fraction_decomposition_l3166_316627

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℝ), x ≠ 1 → x ≠ 4 → x ≠ -2 →
      (x^3 - x - 4) / ((x - 1) * (x - 4) * (x + 2)) =
      A / (x - 1) + B / (x - 4) + C / (x + 2) ∧
      A = 4/9 ∧ B = 28/9 ∧ C = -1/3 := by
sorry

end partial_fraction_decomposition_l3166_316627


namespace cos_alpha_for_given_point_l3166_316646

/-- If the terminal side of angle α passes through the point (√3/2, 1/2), then cos α = √3/2 -/
theorem cos_alpha_for_given_point (α : Real) :
  (∃ (r : Real), r * (Real.sqrt 3 / 2) = Real.cos α ∧ r * (1 / 2) = Real.sin α) →
  Real.cos α = Real.sqrt 3 / 2 := by
sorry

end cos_alpha_for_given_point_l3166_316646


namespace triangle_inequality_l3166_316668

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end triangle_inequality_l3166_316668


namespace tylers_sanctuary_pairs_l3166_316654

/-- Represents the animal sanctuary with three regions -/
structure AnimalSanctuary where
  bird_species : ℕ
  bird_pairs_per_species : ℕ
  marine_species : ℕ
  marine_pairs_per_species : ℕ
  mammal_species : ℕ
  mammal_pairs_per_species : ℕ

/-- Calculates the total number of pairs in the sanctuary -/
def total_pairs (sanctuary : AnimalSanctuary) : ℕ :=
  sanctuary.bird_species * sanctuary.bird_pairs_per_species +
  sanctuary.marine_species * sanctuary.marine_pairs_per_species +
  sanctuary.mammal_species * sanctuary.mammal_pairs_per_species

/-- Theorem stating that the total number of pairs in Tyler's sanctuary is 470 -/
theorem tylers_sanctuary_pairs :
  let tyler_sanctuary : AnimalSanctuary := {
    bird_species := 29,
    bird_pairs_per_species := 7,
    marine_species := 15,
    marine_pairs_per_species := 9,
    mammal_species := 22,
    mammal_pairs_per_species := 6
  }
  total_pairs tyler_sanctuary = 470 := by
  sorry

end tylers_sanctuary_pairs_l3166_316654


namespace john_games_l3166_316683

/-- Calculates the number of unique working games John ended up with -/
def unique_working_games (friend_games : ℕ) (friend_nonworking : ℕ) (garage_games : ℕ) (garage_nonworking : ℕ) (garage_duplicates : ℕ) : ℕ :=
  (friend_games - friend_nonworking) + (garage_games - garage_nonworking - garage_duplicates)

/-- Theorem stating that John ended up with 17 unique working games -/
theorem john_games : unique_working_games 25 12 15 8 3 = 17 := by
  sorry

end john_games_l3166_316683


namespace triangle_perimeter_l3166_316653

theorem triangle_perimeter : 
  let A : ℝ × ℝ := (2, -2)
  let B : ℝ × ℝ := (8, 4)
  let C : ℝ × ℝ := (2, 4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := dist A B + dist B C + dist C A
  perimeter = 12 + 6 * Real.sqrt 2 := by sorry

end triangle_perimeter_l3166_316653


namespace total_texts_is_forty_l3166_316600

/-- The number of texts Sydney sent to Allison and Brittney on both days -/
def total_texts (monday_texts_per_person tuesday_texts_per_person : ℕ) : ℕ :=
  2 * (monday_texts_per_person + tuesday_texts_per_person)

/-- Theorem stating that the total number of texts is 40 -/
theorem total_texts_is_forty :
  total_texts 5 15 = 40 := by
  sorry

end total_texts_is_forty_l3166_316600


namespace age_difference_l3166_316622

/-- Given two people A and B, where B is currently 39 years old, and in 10 years A will be twice as old as B was 10 years ago, this theorem proves that A is currently 9 years older than B. -/
theorem age_difference (A_age B_age : ℕ) : 
  B_age = 39 → 
  A_age + 10 = 2 * (B_age - 10) → 
  A_age - B_age = 9 := by
  sorry

end age_difference_l3166_316622


namespace factorial_divisibility_iff_power_of_two_l3166_316641

theorem factorial_divisibility_iff_power_of_two (n : ℕ) :
  (∃ k : ℕ, n = 2^k) ↔ (2^(n-1) ∣ n!) := by
  sorry

end factorial_divisibility_iff_power_of_two_l3166_316641


namespace valid_param_iff_l3166_316688

/-- A vector parameterization of a line --/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- The line y = 3x - 4 --/
def line (x : ℝ) : ℝ := 3 * x - 4

/-- Predicate for a valid vector parameterization --/
def is_valid_param (p : VectorParam) : Prop :=
  p.y₀ = line p.x₀ ∧ p.dy = 3 * p.dx

/-- Theorem: A vector parameterization is valid iff it satisfies the conditions --/
theorem valid_param_iff (p : VectorParam) :
  is_valid_param p ↔ ∀ t : ℝ, line (p.x₀ + t * p.dx) = p.y₀ + t * p.dy :=
sorry

end valid_param_iff_l3166_316688


namespace circular_table_dice_probability_l3166_316692

/-- The number of people sitting around the circular table -/
def num_people : ℕ := 5

/-- The number of sides on each die -/
def die_sides : ℕ := 8

/-- The probability that two adjacent people roll different numbers -/
def prob_different_adjacent : ℚ := 7 / 8

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ := (prob_different_adjacent) ^ num_people

theorem circular_table_dice_probability :
  prob_no_adjacent_same = (7 / 8) ^ 5 :=
sorry

end circular_table_dice_probability_l3166_316692


namespace total_water_consumption_l3166_316630

def traveler_ounces : ℕ := 32
def camel_multiplier : ℕ := 7
def ounces_per_gallon : ℕ := 128

theorem total_water_consumption :
  (traveler_ounces + camel_multiplier * traveler_ounces) / ounces_per_gallon = 2 := by
  sorry

end total_water_consumption_l3166_316630


namespace equation_solutions_l3166_316672

theorem equation_solutions (m : ℕ+) :
  let f := fun (x y : ℕ) => x^2 + y^2 + 2*x*y - m*x - m*y - m - 1
  (∃! s : Finset (ℕ × ℕ), s.card = m ∧ 
    ∀ (p : ℕ × ℕ), p ∈ s ↔ (f p.1 p.2 = 0 ∧ p.1 > 0 ∧ p.2 > 0)) := by
  sorry

end equation_solutions_l3166_316672


namespace smallest_valid_arrangement_l3166_316624

def is_valid_arrangement (n : ℕ) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ : ℕ),
    a₁ = 15 ∧
    a₂ = n ∧
    a₃ = 1 ∧
    a₄ = 6 ∧
    n % a₁ = 0 ∧
    n % a₂ = 0 ∧
    n % a₃ = 0 ∧
    n % a₄ = 0 ∧
    ∀ (i j : ℕ), i ≠ j → (n / a₁) ≠ (n / a₂) ∧ (n / a₁) ≠ (n / a₃) ∧ (n / a₁) ≠ (n / a₄) ∧
                         (n / a₂) ≠ (n / a₃) ∧ (n / a₂) ≠ (n / a₄) ∧ (n / a₃) ≠ (n / a₄)

theorem smallest_valid_arrangement : 
  ∃ (n : ℕ), is_valid_arrangement n ∧ ∀ (m : ℕ), m < n → ¬is_valid_arrangement m :=
by sorry

end smallest_valid_arrangement_l3166_316624


namespace phone_number_probability_l3166_316606

theorem phone_number_probability : 
  ∀ (n : ℕ) (p : ℚ),
    n = 10 →  -- There are 10 possible digits
    p = 1 / n →  -- Probability of correct guess on each attempt
    (p + p + p) = 3 / 10  -- Probability of success in no more than 3 attempts
    := by sorry

end phone_number_probability_l3166_316606


namespace sqrt_plus_square_zero_implies_diff_l3166_316684

theorem sqrt_plus_square_zero_implies_diff (a b : ℝ) : 
  Real.sqrt (a - 3) + (b + 1)^2 = 0 → a - b = 4 := by
  sorry

end sqrt_plus_square_zero_implies_diff_l3166_316684


namespace product_and_reciprocal_sum_l3166_316632

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 12 → (1 / x) = 3 * (1 / y) → x + y = 8 := by
  sorry

end product_and_reciprocal_sum_l3166_316632


namespace quadratic_expression_values_l3166_316634

theorem quadratic_expression_values (m n : ℤ) 
  (hm : |m| = 3) 
  (hn : |n| = 2) 
  (hmn : m < n) : 
  m^2 + m*n + n^2 = 7 ∨ m^2 + m*n + n^2 = 19 := by
sorry

end quadratic_expression_values_l3166_316634


namespace part1_part2_l3166_316681

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - a + 1) ≤ 0}
def B : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define the proposition p
def p (m : ℝ) : Prop := ∃ x ∈ B, x^2 + (2*m + 1)*x + m^2 - m > 8

-- Theorem for part 1
theorem part1 : 
  (∀ x, x ∈ A a → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A a) → 
  a > -1 ∧ a < 1 :=
sorry

-- Theorem for part 2
theorem part2 : 
  (¬ p m) → m ≥ -1 ∧ m ≤ 2 :=
sorry

end part1_part2_l3166_316681


namespace g_neg_three_l3166_316695

def g (x : ℝ) : ℝ := 10 * x^3 - 4 * x^2 - 6 * x + 7

theorem g_neg_three : g (-3) = -281 := by
  sorry

end g_neg_three_l3166_316695


namespace computer_price_after_15_years_l3166_316665

/-- The price of a computer after a certain number of 5-year periods, given an initial price and a price decrease rate. -/
def computer_price (initial_price : ℝ) (decrease_rate : ℝ) (periods : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ periods

/-- Theorem stating that a computer with an initial price of 8100 yuan and a price decrease of 1/3 every 5 years will cost 2400 yuan after 15 years. -/
theorem computer_price_after_15_years :
  computer_price 8100 (1/3) 3 = 2400 := by
  sorry

end computer_price_after_15_years_l3166_316665


namespace stating_three_plane_division_l3166_316616

/-- Represents the possible numbers of parts that three planes can divide 3D space into -/
inductive PlaneDivision : Type
  | four : PlaneDivision
  | six : PlaneDivision
  | seven : PlaneDivision
  | eight : PlaneDivision

/-- Represents a configuration of three planes in 3D space -/
structure ThreePlaneConfiguration where
  -- Add necessary fields to represent the configuration

/-- 
Given a configuration of three planes in 3D space, 
returns the number of parts the space is divided into
-/
def countParts (config : ThreePlaneConfiguration) : PlaneDivision :=
  sorry

/-- 
Theorem stating that three planes can only divide 3D space into 4, 6, 7, or 8 parts,
and all these cases are possible
-/
theorem three_plane_division :
  (∀ config : ThreePlaneConfiguration, ∃ n : PlaneDivision, countParts config = n) ∧
  (∀ n : PlaneDivision, ∃ config : ThreePlaneConfiguration, countParts config = n) :=
sorry

end stating_three_plane_division_l3166_316616


namespace tangent_line_b_value_l3166_316621

/-- The curve y = x^3 + ax + 1 passes through the point (2, 3) -/
def curve_passes_through (a : ℝ) : Prop :=
  2^3 + a*2 + 1 = 3

/-- The derivative of the curve y = x^3 + ax + 1 -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ :=
  3*x^2 + a

/-- The line y = kx + b is tangent to the curve y = x^3 + ax + 1 at x = 2 -/
def line_tangent_to_curve (a k b : ℝ) : Prop :=
  k = curve_derivative a 2

/-- The line y = kx + b passes through the point (2, 3) -/
def line_passes_through (k b : ℝ) : Prop :=
  k*2 + b = 3

theorem tangent_line_b_value (a k b : ℝ) :
  curve_passes_through a →
  line_tangent_to_curve a k b →
  line_passes_through k b →
  b = -15 := by
  sorry

end tangent_line_b_value_l3166_316621


namespace banana_groups_l3166_316698

theorem banana_groups (total_bananas : ℕ) (num_groups : ℕ) 
  (h1 : total_bananas = 392) 
  (h2 : num_groups = 196) : 
  total_bananas / num_groups = 2 := by
  sorry

end banana_groups_l3166_316698


namespace divisibility_condition_l3166_316677

theorem divisibility_condition (n p : ℕ) (h_prime : Nat.Prime p) (h_bound : n < 2 * p) :
  (((p - 1) ^ n + 1) % (n ^ (p - 1)) = 0) ↔ 
  ((n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
sorry

end divisibility_condition_l3166_316677


namespace complex_division_simplification_l3166_316643

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  (-3 + 2*i) / (1 + i) = -1/2 + 5/2*i := by sorry

end complex_division_simplification_l3166_316643


namespace total_flowers_in_gardens_l3166_316642

/-- Given 10 gardens, each with 544 pots, and each pot containing 32 flowers,
    prove that the total number of flowers in all gardens is 174,080. -/
theorem total_flowers_in_gardens : 
  let num_gardens : ℕ := 10
  let pots_per_garden : ℕ := 544
  let flowers_per_pot : ℕ := 32
  num_gardens * pots_per_garden * flowers_per_pot = 174080 :=
by sorry

end total_flowers_in_gardens_l3166_316642


namespace monogram_count_l3166_316602

/-- The number of letters in the alphabet before 'G' -/
def letters_before_g : Nat := 6

/-- The number of letters in the alphabet after 'G' -/
def letters_after_g : Nat := 18

/-- The total number of possible monograms with 'G' as the middle initial,
    and the other initials different and in alphabetical order -/
def total_monograms : Nat := letters_before_g * letters_after_g

theorem monogram_count :
  total_monograms = 108 := by
  sorry

end monogram_count_l3166_316602


namespace inequality_system_solvability_l3166_316607

theorem inequality_system_solvability (n : ℕ) : 
  (∃ x : ℝ, 
    (1 < x ∧ x < 2) ∧
    (2 < x^2 ∧ x^2 < 3) ∧
    (∀ k : ℕ, 3 ≤ k ∧ k ≤ n → k < x^k ∧ x^k < k + 1)) ↔
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
by sorry

end inequality_system_solvability_l3166_316607


namespace geometric_sequence_ratio_l3166_316608

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_sum_24 : a 2 + a 4 = 20)
  (h_sum_35 : a 3 + a 5 = 40) :
  q = 2 := by
sorry

end geometric_sequence_ratio_l3166_316608


namespace erased_digit_greater_than_original_l3166_316676

-- Define the fraction
def fraction : Rat := 3 / 7

-- Define the number of digits after the decimal point
def num_digits : Nat := 1000

-- Define the position of the digit to be erased
def erased_position : Nat := 500

-- Function to get the nth digit after the decimal point
def nth_digit (n : Nat) : Nat :=
  sorry

-- Function to construct the number after erasing the 500th digit
def number_after_erasing : Rat :=
  sorry

-- Theorem statement
theorem erased_digit_greater_than_original :
  number_after_erasing > fraction :=
sorry

end erased_digit_greater_than_original_l3166_316676


namespace least_positive_integer_with_remainders_l3166_316656

theorem least_positive_integer_with_remainders : ∃! x : ℕ,
  x > 0 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 7 = 6 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 7 = 6 → x ≤ y :=
by
  use 139
  sorry

end least_positive_integer_with_remainders_l3166_316656


namespace small_portion_visible_implies_intersection_l3166_316663

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is above a line -/
def isAboveLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c > 0

/-- Predicate to check if a circle intersects a line -/
def circleIntersectsLine (c : Circle) (l : Line) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
                 l.a * p.1 + l.b * p.2 + l.c = 0

/-- Predicate to check if a small portion of a circle is visible above a line -/
def smallPortionVisible (c : Circle) (l : Line) : Prop :=
  ∃ (p q : ℝ × ℝ), 
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
    isAboveLine p l ∧
    isAboveLine q l ∧
    ∀ (r : ℝ × ℝ), (r.1 - c.center.1)^2 + (r.2 - c.center.2)^2 = c.radius^2 →
                   isAboveLine r l →
                   (r.1 ≥ min p.1 q.1 ∧ r.1 ≤ max p.1 q.1) ∧
                   (r.2 ≥ min p.2 q.2 ∧ r.2 ≤ max p.2 q.2)

theorem small_portion_visible_implies_intersection (c : Circle) (l : Line) :
  smallPortionVisible c l → circleIntersectsLine c l :=
by sorry

end small_portion_visible_implies_intersection_l3166_316663


namespace regression_line_equation_l3166_316674

/-- Given a slope and a point on a line, calculate the y-intercept -/
def calculate_y_intercept (slope : ℝ) (point : ℝ × ℝ) : ℝ :=
  point.2 - slope * point.1

/-- The regression line problem -/
theorem regression_line_equation (slope : ℝ) (point : ℝ × ℝ) 
  (h_slope : slope = 1.23)
  (h_point : point = (4, 5)) :
  calculate_y_intercept slope point = 0.08 := by
  sorry

end regression_line_equation_l3166_316674


namespace complex_power_magnitude_l3166_316687

theorem complex_power_magnitude : Complex.abs ((2 + 2*Complex.I)^6) = 512 := by
  sorry

end complex_power_magnitude_l3166_316687


namespace min_value_polynomial_l3166_316680

theorem min_value_polynomial (x : ℝ) : 
  (∃ (m : ℝ), ∀ (x : ℝ), (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 ≥ m) ∧ 
  (∃ (x : ℝ), (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 = 2022) := by
  sorry

end min_value_polynomial_l3166_316680


namespace integral_x_squared_plus_4x_plus_3_cos_x_l3166_316639

theorem integral_x_squared_plus_4x_plus_3_cos_x : 
  ∫ (x : ℝ) in (-1)..0, (x^2 + 4*x + 3) * Real.cos x = 4 - 2 * Real.cos 1 - 2 * Real.sin 1 := by
  sorry

end integral_x_squared_plus_4x_plus_3_cos_x_l3166_316639


namespace exists_solution_for_calendar_equation_l3166_316652

theorem exists_solution_for_calendar_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end exists_solution_for_calendar_equation_l3166_316652


namespace smallest_cube_for_pyramid_l3166_316659

/-- Represents a triangular pyramid with a given height and base side length -/
structure TriangularPyramid where
  height : ℝ
  baseSide : ℝ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ :=
  c.sideLength ^ 3

/-- Checks if a cube can contain a triangular pyramid upright -/
def canContainPyramid (c : Cube) (p : TriangularPyramid) : Prop :=
  c.sideLength ≥ p.height ∧ c.sideLength ≥ p.baseSide

/-- The main theorem statement -/
theorem smallest_cube_for_pyramid (p : TriangularPyramid)
    (h1 : p.height = 15)
    (h2 : p.baseSide = 12) :
    ∃ (c : Cube), canContainPyramid c p ∧
    cubeVolume c = 3375 ∧
    ∀ (c' : Cube), canContainPyramid c' p → cubeVolume c' ≥ cubeVolume c := by
  sorry

end smallest_cube_for_pyramid_l3166_316659


namespace initial_honey_amount_honey_jar_problem_l3166_316651

/-- The amount of honey remaining after each extraction -/
def honey_remaining (initial_honey : ℝ) (num_extractions : ℕ) : ℝ :=
  initial_honey * (0.8 ^ num_extractions)

/-- Theorem stating the initial amount of honey given the final amount and number of extractions -/
theorem initial_honey_amount 
  (final_honey : ℝ) 
  (num_extractions : ℕ) 
  (h_final : honey_remaining initial_honey num_extractions = final_honey) : 
  initial_honey = final_honey / (0.8 ^ num_extractions) :=
by sorry

/-- The solution to the honey jar problem -/
theorem honey_jar_problem : 
  ∃ (initial_honey : ℝ), 
    honey_remaining initial_honey 4 = 512 ∧ 
    initial_honey = 1250 :=
by sorry

end initial_honey_amount_honey_jar_problem_l3166_316651


namespace perpendicular_line_equation_l3166_316613

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation 
  (l : Line)
  (A : Point)
  (given_line : Line)
  (h1 : A.liesOn l)
  (h2 : l.perpendicular given_line)
  (h3 : A.x = -1)
  (h4 : A.y = 3)
  (h5 : given_line.a = 1)
  (h6 : given_line.b = -2)
  (h7 : given_line.c = -3) :
  l.a = 2 ∧ l.b = 1 ∧ l.c = -1 := by
  sorry

end perpendicular_line_equation_l3166_316613


namespace point_in_first_quadrant_l3166_316617

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def is_in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point (3,5) -/
def given_point : Point :=
  { x := 3, y := 5 }

/-- Theorem: The given point (3,5) is in the first quadrant -/
theorem point_in_first_quadrant :
  is_in_first_quadrant given_point := by
  sorry

end point_in_first_quadrant_l3166_316617


namespace range_of_m_l3166_316640

theorem range_of_m (m : ℝ) : 
  (∀ (x y : ℝ), x > 0 → y > 0 → (2*x - y/Real.exp 1) * Real.log (y/x) ≤ x/(m*Real.exp 1)) ↔ 
  (m > 0 ∧ m ≤ 1/Real.exp 1) :=
sorry

end range_of_m_l3166_316640


namespace thermometer_price_is_two_l3166_316604

/-- Represents the sales data for thermometers and hot-water bottles --/
structure SalesData where
  thermometer_price : ℝ
  hotwater_bottle_price : ℝ
  total_sales : ℝ
  thermometer_to_bottle_ratio : ℕ
  bottles_sold : ℕ

/-- Theorem stating that the thermometer price is 2 dollars given the sales data --/
theorem thermometer_price_is_two (data : SalesData)
  (h1 : data.hotwater_bottle_price = 6)
  (h2 : data.total_sales = 1200)
  (h3 : data.thermometer_to_bottle_ratio = 7)
  (h4 : data.bottles_sold = 60)
  : data.thermometer_price = 2 := by
  sorry


end thermometer_price_is_two_l3166_316604


namespace hexagons_in_100th_ring_hexagons_in_nth_ring_formula_l3166_316682

/-- Represents the number of hexagons in the nth ring of a hexagonal array -/
def hexagons_in_nth_ring (n : ℕ) : ℕ := 6 * n

/-- The hexagonal array satisfies the initial conditions -/
axiom first_ring : hexagons_in_nth_ring 1 = 6
axiom second_ring : hexagons_in_nth_ring 2 = 12

/-- Theorem: The number of hexagons in the 100th ring is 600 -/
theorem hexagons_in_100th_ring : hexagons_in_nth_ring 100 = 600 := by
  sorry

/-- Theorem: The number of hexagons in the nth ring is 6n -/
theorem hexagons_in_nth_ring_formula (n : ℕ) : hexagons_in_nth_ring n = 6 * n := by
  sorry

end hexagons_in_100th_ring_hexagons_in_nth_ring_formula_l3166_316682


namespace married_men_fraction_l3166_316614

theorem married_men_fraction (total_women : ℕ) (total_people : ℕ) 
  (h_women_positive : total_women > 0)
  (h_total_positive : total_people > 0)
  (h_single_prob : (3 : ℚ) / 7 = (total_women - (total_people - total_women)) / total_women) :
  (total_people - total_women) / total_people = (4 : ℚ) / 11 := by
sorry

end married_men_fraction_l3166_316614


namespace simplify_expression_l3166_316610

theorem simplify_expression : 
  (Real.sqrt 392 / Real.sqrt 336) + (Real.sqrt 200 / Real.sqrt 128) + 1 = 41 / 12 := by
  sorry

end simplify_expression_l3166_316610


namespace flour_to_add_l3166_316623

/-- Given a cake recipe and partially added ingredients, calculate the remaining amount to be added -/
theorem flour_to_add (total_required : ℕ) (already_added : ℕ) (h : total_required ≥ already_added) :
  total_required - already_added = 8 - 4 :=
by
  sorry

#check flour_to_add

end flour_to_add_l3166_316623


namespace geometric_arithmetic_sequence_ratio_l3166_316662

theorem geometric_arithmetic_sequence_ratio (x y z : ℝ) 
  (h1 : (4*y)^2 = (3*x)*(5*z))  -- Geometric sequence condition
  (h2 : 2/y = 1/x + 1/z)        -- Arithmetic sequence condition
  : x/z + z/x = 34/15 := by
  sorry

end geometric_arithmetic_sequence_ratio_l3166_316662


namespace expression_value_l3166_316635

theorem expression_value : (16.25 / 0.25) + (8.4 / 3) - (0.75 / 0.05) = 52.8 := by
  sorry

end expression_value_l3166_316635


namespace area_of_inscribed_square_l3166_316678

/-- The area of a square inscribed in a circle, which is itself inscribed in an equilateral triangle with side length 6 cm. -/
theorem area_of_inscribed_square (triangle_side : ℝ) (h : triangle_side = 6) : 
  let circle_radius := triangle_side / (2 * Real.sqrt 3)
  let square_side := 2 * circle_radius / Real.sqrt 2
  square_side ^ 2 = 6 * (3 - Real.sqrt 3) := by sorry

end area_of_inscribed_square_l3166_316678


namespace building_shadow_length_l3166_316671

/-- Given a flagpole and a building under similar conditions, 
    calculate the length of the shadow cast by the building -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : flagpole_shadow = 45)
  (h3 : building_height = 26)
  : ∃ building_shadow : ℝ, building_shadow = 65 := by
  sorry

#check building_shadow_length

end building_shadow_length_l3166_316671


namespace spider_journey_l3166_316675

theorem spider_journey (r : ℝ) (third_leg : ℝ) (h1 : r = 50) (h2 : third_leg = 70) :
  let diameter := 2 * r
  let second_leg := Real.sqrt (diameter^2 - third_leg^2)
  diameter + third_leg + second_leg = 170 + Real.sqrt 5100 := by
sorry

end spider_journey_l3166_316675


namespace num_divisors_8_factorial_is_96_l3166_316686

/-- The number of positive divisors of 8! -/
def num_divisors_8_factorial : ℕ :=
  let factorial_8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  -- Definition of the number of divisors function not provided, so we'll declare it
  sorry

/-- Theorem: The number of positive divisors of 8! is 96 -/
theorem num_divisors_8_factorial_is_96 :
  num_divisors_8_factorial = 96 := by
  sorry

end num_divisors_8_factorial_is_96_l3166_316686


namespace triangle_base_length_l3166_316696

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 6 → height = 4 → area = (base * height) / 2 → base = 3 := by
  sorry

end triangle_base_length_l3166_316696


namespace ashley_tablet_battery_life_l3166_316620

/-- Represents the battery life of Ashley's tablet -/
structure TabletBattery where
  fullLifeIdle : ℝ  -- Battery life in hours when idle
  fullLifeActive : ℝ  -- Battery life in hours when active
  usedTime : ℝ  -- Total time used since last charge
  activeTime : ℝ  -- Time spent actively using the tablet

/-- Calculates the remaining battery life of Ashley's tablet -/
def remainingBatteryLife (tb : TabletBattery) : ℝ :=
  sorry

/-- Theorem stating that Ashley's tablet will last 8 more hours -/
theorem ashley_tablet_battery_life :
  ∀ (tb : TabletBattery),
    tb.fullLifeIdle = 36 ∧
    tb.fullLifeActive = 4 ∧
    tb.usedTime = 12 ∧
    tb.activeTime = 2 →
    remainingBatteryLife tb = 8 :=
  sorry

end ashley_tablet_battery_life_l3166_316620


namespace solve_journey_problem_l3166_316690

def journey_problem (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : Prop :=
  let half_distance := (total_time * speed1 * speed2) / (speed1 + speed2)
  total_time = half_distance / speed1 + half_distance / speed2 →
  2 * half_distance = 240

theorem solve_journey_problem :
  journey_problem 20 10 15 := by
  sorry

end solve_journey_problem_l3166_316690


namespace smallest_divisible_k_l3166_316666

/-- The polynomial p(z) = z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1 -/
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

/-- The function f(k) = z^k - 1 -/
def f (k : ℕ) (z : ℂ) : ℂ := z^k - 1

/-- Theorem: The smallest positive integer k such that p(z) divides f(k)(z) is 112 -/
theorem smallest_divisible_k : (∀ z : ℂ, p z ∣ f 112 z) ∧
  (∀ k : ℕ, k < 112 → ∃ z : ℂ, ¬(p z ∣ f k z)) :=
sorry

end smallest_divisible_k_l3166_316666


namespace tan_pi_plus_2alpha_l3166_316699

theorem tan_pi_plus_2alpha (α : Real) 
  (h1 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi))
  (h2 : Real.sin (Real.pi / 2 + α) = 1 / 3) : 
  Real.tan (Real.pi + 2 * α) = 4 * Real.sqrt 2 / 7 := by
  sorry

end tan_pi_plus_2alpha_l3166_316699


namespace simplify_and_rationalize_l3166_316605

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 6 / Real.sqrt 8) =
  (3 * Real.sqrt 385) / 154 := by
  sorry

end simplify_and_rationalize_l3166_316605


namespace lost_ship_depth_l3166_316629

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
theorem lost_ship_depth (rate : ℝ) (time : ℝ) (h1 : rate = 32) (h2 : time = 200) :
  rate * time = 6400 := by
  sorry

end lost_ship_depth_l3166_316629


namespace correct_shirt_price_l3166_316637

-- Define the price of one shirt
def shirt_price : ℝ := 10

-- Define the cost of two shirts
def cost_two_shirts (p : ℝ) : ℝ := 1.5 * p

-- Define the cost of three shirts
def cost_three_shirts (p : ℝ) : ℝ := 1.9 * p

-- Define the savings when buying three shirts
def savings_three_shirts (p : ℝ) : ℝ := 3 * p - cost_three_shirts p

-- Theorem stating that the shirt price is correct
theorem correct_shirt_price :
  cost_two_shirts shirt_price = 1.5 * shirt_price ∧
  cost_three_shirts shirt_price = 1.9 * shirt_price ∧
  savings_three_shirts shirt_price = 11 :=
by sorry

end correct_shirt_price_l3166_316637


namespace peters_pizza_fraction_l3166_316664

-- Define the number of slices in the pizza
def total_slices : ℕ := 16

-- Define the number of whole slices Peter ate
def whole_slices_eaten : ℕ := 1

-- Define the number of slices shared
def shared_slices : ℕ := 2

-- Theorem statement
theorem peters_pizza_fraction :
  (whole_slices_eaten : ℚ) / total_slices + 
  (shared_slices : ℚ) / total_slices / 2 = 1 / 8 := by
  sorry


end peters_pizza_fraction_l3166_316664


namespace yoongi_initial_books_l3166_316611

/-- Represents the number of books each person has -/
structure BookCount where
  yoongi : ℕ
  eunji : ℕ
  yuna : ℕ

/-- Represents the book exchange described in the problem -/
def exchange (initial : BookCount) : BookCount :=
  { yoongi := initial.yoongi - 5 + 15,
    eunji := initial.eunji + 5 - 10,
    yuna := initial.yuna + 10 - 15 }

/-- Theorem stating that if after the exchange all have 45 books, 
    Yoongi must have started with 35 books -/
theorem yoongi_initial_books 
  (initial : BookCount) 
  (h : exchange initial = {yoongi := 45, eunji := 45, yuna := 45}) : 
  initial.yoongi = 35 := by
  sorry

end yoongi_initial_books_l3166_316611


namespace line_length_after_erasing_l3166_316660

-- Define the original length in meters
def original_length_m : ℝ := 1.5

-- Define the erased length in centimeters
def erased_length_cm : ℝ := 37.5

-- Define the conversion factor from meters to centimeters
def m_to_cm : ℝ := 100

-- Theorem statement
theorem line_length_after_erasing :
  (original_length_m * m_to_cm - erased_length_cm) = 112.5 := by
  sorry

end line_length_after_erasing_l3166_316660


namespace largest_prime_to_test_primality_l3166_316670

theorem largest_prime_to_test_primality (n : ℕ) : 
  900 ≤ n ∧ n ≤ 950 → 
  (∀ (p : ℕ), p.Prime → p ≤ 29 → (p ∣ n ↔ ¬n.Prime)) ∧
  (∀ (p : ℕ), p.Prime → p > 29 → (p ∣ n → ¬n.Prime)) :=
sorry

end largest_prime_to_test_primality_l3166_316670


namespace equilateral_triangles_congruence_l3166_316649

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Two triangles are congruent if all their corresponding sides are equal -/
def congruent (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side = t2.side

theorem equilateral_triangles_congruence (t1 t2 : EquilateralTriangle) :
  congruent t1 t2 ↔ t1.side = t2.side :=
sorry

end equilateral_triangles_congruence_l3166_316649


namespace probability_at_least_one_girl_pair_l3166_316647

/-- The number of ways to split 2n people into n pairs --/
def total_pairings (n : ℕ) : ℕ := (2 * n).factorial / (2^n * n.factorial)

/-- The number of ways to pair n boys with n girls --/
def boy_girl_pairings (n : ℕ) : ℕ := n.factorial

theorem probability_at_least_one_girl_pair (n : ℕ) (hn : n = 5) :
  (total_pairings n - boy_girl_pairings n) / total_pairings n = 23640 / 23760 :=
sorry

end probability_at_least_one_girl_pair_l3166_316647


namespace equation_solution_l3166_316644

theorem equation_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 := by
  sorry

end equation_solution_l3166_316644


namespace quadratic_function_range_l3166_316633

theorem quadratic_function_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 - 4 * x₁ - 13 = 0 ∧ a * x₂^2 - 4 * x₂ - 13 = 0) →
  (a > -4/13 ∧ a ≠ 0) :=
by sorry

end quadratic_function_range_l3166_316633


namespace one_third_percent_of_180_l3166_316689

theorem one_third_percent_of_180 : (1 / 3) * (1 / 100) * 180 = 0.6 := by
  sorry

end one_third_percent_of_180_l3166_316689
