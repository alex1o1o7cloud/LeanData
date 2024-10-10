import Mathlib

namespace fraction_inequality_l3996_399688

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end fraction_inequality_l3996_399688


namespace age_difference_is_four_l3996_399621

/-- The difference between the ages of Albert's parents -/
def age_difference (albert_age : ℕ) : ℕ :=
  let father_age := albert_age + 48
  let brother_age := albert_age - 2
  let mother_age := brother_age + 46
  father_age - mother_age

/-- Theorem stating that the difference between the ages of Albert's parents is 4 years -/
theorem age_difference_is_four (albert_age : ℕ) (h : albert_age ≥ 2) :
  age_difference albert_age = 4 := by
  sorry

end age_difference_is_four_l3996_399621


namespace daniels_driving_speed_l3996_399615

/-- Proves that given the conditions of Daniel's driving scenario, 
    the speed for the first 32 miles on Monday is 2x miles per hour. -/
theorem daniels_driving_speed (x : ℝ) (y : ℝ) : 
  x > 0 → -- Ensure x is positive for valid speed
  96 / x = (32 / y) + (128 / x) * (3/2) →
  y = 2 * x :=
by sorry

end daniels_driving_speed_l3996_399615


namespace recipe_sugar_requirement_l3996_399648

/-- The number of cups of sugar Mary has already added to the cake. -/
def sugar_added : ℕ := 10

/-- The number of cups of sugar Mary still needs to add to the cake. -/
def sugar_to_add : ℕ := 1

/-- The total number of cups of sugar required by the recipe. -/
def total_sugar : ℕ := sugar_added + sugar_to_add

/-- The number of cups of flour required by the recipe. -/
def flour_required : ℕ := 9

/-- The number of cups of flour Mary has already added to the cake. -/
def flour_added : ℕ := 12

theorem recipe_sugar_requirement :
  total_sugar = 11 := by sorry

end recipe_sugar_requirement_l3996_399648


namespace sufficient_not_necessary_condition_l3996_399631

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, |a + b| > 1 → |a| + |b| > 1) ∧
  (∃ a b : ℝ, |a| + |b| > 1 ∧ |a + b| ≤ 1) :=
by sorry

end sufficient_not_necessary_condition_l3996_399631


namespace double_inequality_solution_l3996_399656

theorem double_inequality_solution (x : ℝ) : 
  (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 1) ↔ (1 < x ∧ x < 3) := by
  sorry

end double_inequality_solution_l3996_399656


namespace expression_value_l3996_399669

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 3) :
  c * d + m - (a + b) / m = 4 ∨ c * d + m - (a + b) / m = -2 := by
sorry

end expression_value_l3996_399669


namespace no_cubic_linear_terms_implies_value_l3996_399644

theorem no_cubic_linear_terms_implies_value (m n : ℝ) :
  (∀ x : ℝ, m * x^3 - 2 * x^2 + 3 * x - 4 * x^3 + 5 * x^2 - n * x = 3 * x^2) →
  m^2 - 2 * m * n + n^2 = 1 := by
  sorry

end no_cubic_linear_terms_implies_value_l3996_399644


namespace prob_two_red_without_replacement_prob_two_red_with_replacement_prob_at_least_one_red_with_replacement_l3996_399642

def total_balls : ℕ := 5
def red_balls : ℕ := 3
def white_balls : ℕ := 2

/-- Probability of drawing exactly 2 red balls without replacement -/
theorem prob_two_red_without_replacement :
  (Nat.choose red_balls 2 : ℚ) / (Nat.choose total_balls 2) = 3 / 10 := by sorry

/-- Probability of drawing exactly 2 red balls with replacement -/
theorem prob_two_red_with_replacement :
  (red_balls : ℚ) / total_balls * (red_balls : ℚ) / total_balls = 9 / 25 := by sorry

/-- Probability of drawing at least 1 red ball with replacement -/
theorem prob_at_least_one_red_with_replacement :
  1 - ((white_balls : ℚ) / total_balls) ^ 2 = 21 / 25 := by sorry

end prob_two_red_without_replacement_prob_two_red_with_replacement_prob_at_least_one_red_with_replacement_l3996_399642


namespace complex_sum_l3996_399677

theorem complex_sum (z : ℂ) (h : z = (1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2)) :
  z + 2*z^2 + 3*z^3 + 4*z^4 + 5*z^5 + 6*z^6 = 3 - Complex.I * 3 * Real.sqrt 3 := by
  sorry

end complex_sum_l3996_399677


namespace geoff_total_spending_l3996_399671

/-- Geoff's spending on sneakers over three days -/
def geoff_spending (x : ℝ) : ℝ := x + 4*x + 5*x

/-- Theorem: Geoff's total spending over three days equals 10 times his Monday spending -/
theorem geoff_total_spending (x : ℝ) : geoff_spending x = 10 * x := by
  sorry

end geoff_total_spending_l3996_399671


namespace farthest_vertex_distance_l3996_399612

/-- Given a rectangle ABCD with area 48 and diagonal 10, and a point O such that OB = OD = 13,
    the distance from O to the farthest vertex of the rectangle is 7√(29/5). -/
theorem farthest_vertex_distance (A B C D O : ℝ × ℝ) : 
  let area := abs ((B.1 - A.1) * (D.2 - A.2))
  let diagonal := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let OB_dist := Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2)
  let OD_dist := Real.sqrt ((O.1 - D.1)^2 + (O.2 - D.2)^2)
  area = 48 ∧ diagonal = 10 ∧ OB_dist = 13 ∧ OD_dist = 13 →
  max (Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2))
      (max (Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2))
           (max (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2))
                (Real.sqrt ((O.1 - D.1)^2 + (O.2 - D.2)^2))))
  = 7 * Real.sqrt (29/5) :=
by
  sorry


end farthest_vertex_distance_l3996_399612


namespace units_digit_of_n_l3996_399643

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_n (m n : ℕ) :
  m * n = 23^5 →
  units_digit m = 4 →
  units_digit n = 8 :=
by
  sorry

end units_digit_of_n_l3996_399643


namespace fuelUsageTheorem_l3996_399627

/-- Calculates the total fuel usage over four weeks given initial usage and percentage changes -/
def totalFuelUsage (initialUsage : ℝ) (week2Change : ℝ) (week3Change : ℝ) (week4Change : ℝ) : ℝ :=
  let week1 := initialUsage
  let week2 := week1 * (1 + week2Change)
  let week3 := week2 * (1 - week3Change)
  let week4 := week3 * (1 + week4Change)
  week1 + week2 + week3 + week4

/-- Theorem stating that the total fuel usage over four weeks is 94.85 gallons -/
theorem fuelUsageTheorem :
  totalFuelUsage 25 0.1 0.3 0.2 = 94.85 := by
  sorry

end fuelUsageTheorem_l3996_399627


namespace quadratic_minimum_l3996_399645

/-- The quadratic function f(x) = x^2 + 2px + p^2 -/
def f (p : ℝ) (x : ℝ) : ℝ := x^2 + 2*p*x + p^2

theorem quadratic_minimum (p : ℝ) (hp : p > 0) (hp2 : 2*p + p^2 = 10) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f p x_min ≤ f p x ∧ x_min = -2 :=
sorry

end quadratic_minimum_l3996_399645


namespace bridge_length_l3996_399685

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 180 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 195 := by
sorry

end bridge_length_l3996_399685


namespace hilt_bee_count_l3996_399653

/-- The number of bees Mrs. Hilt saw on the first day -/
def first_day_bees : ℕ := 144

/-- The multiplier for the number of bees on the second day -/
def bee_multiplier : ℕ := 3

/-- The number of bees Mrs. Hilt saw on the second day -/
def second_day_bees : ℕ := first_day_bees * bee_multiplier

theorem hilt_bee_count : second_day_bees = 432 := by
  sorry

end hilt_bee_count_l3996_399653


namespace circular_segment_area_l3996_399664

theorem circular_segment_area (r a : ℝ) (hr : r > 0) (ha : 0 < a ∧ a < 2*r) :
  let segment_area := r^2 * Real.arcsin (a / (2*r)) - (a/4) * Real.sqrt (4*r^2 - a^2)
  segment_area = r^2 * Real.arcsin (a / (2*r)) - (a/4) * Real.sqrt (4*r^2 - a^2) :=
by sorry

end circular_segment_area_l3996_399664


namespace households_with_car_l3996_399678

theorem households_with_car (total : ℕ) (without_car_or_bike : ℕ) (with_both : ℕ) (with_bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : without_car_or_bike = 11)
  (h3 : with_both = 16)
  (h4 : with_bike_only = 35) :
  total - without_car_or_bike - with_bike_only + with_both = 60 := by
  sorry

#check households_with_car

end households_with_car_l3996_399678


namespace divisible_by_two_l3996_399665

theorem divisible_by_two (m n : ℕ) : 
  2 ∣ (5*m + n + 1) * (3*m - n + 4) := by
  sorry

end divisible_by_two_l3996_399665


namespace sum_of_products_l3996_399632

theorem sum_of_products (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 5*x₂ + 10*x₃ + 17*x₄ + 26*x₅ + 37*x₆ + 50*x₇ + 65*x₈ = 2)
  (eq2 : 5*x₁ + 10*x₂ + 17*x₃ + 26*x₄ + 37*x₅ + 50*x₆ + 65*x₇ + 82*x₈ = 14)
  (eq3 : 10*x₁ + 17*x₂ + 26*x₃ + 37*x₄ + 50*x₅ + 65*x₆ + 82*x₇ + 101*x₈ = 140) :
  17*x₁ + 26*x₂ + 37*x₃ + 50*x₄ + 65*x₅ + 82*x₆ + 101*x₇ + 122*x₈ = 608 := by
  sorry

end sum_of_products_l3996_399632


namespace tan_product_simplification_l3996_399623

theorem tan_product_simplification :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
  sorry

end tan_product_simplification_l3996_399623


namespace prob_at_most_one_girl_l3996_399630

/-- The probability of selecting at most one girl when randomly choosing 3 people
    from a group of 4 boys and 2 girls is 4/5. -/
theorem prob_at_most_one_girl (total : Nat) (boys : Nat) (girls : Nat) (selected : Nat) :
  total = boys + girls →
  boys = 4 →
  girls = 2 →
  selected = 3 →
  (Nat.choose boys selected + Nat.choose boys (selected - 1) * Nat.choose girls 1) /
    Nat.choose total selected = 4 / 5 := by
  sorry

end prob_at_most_one_girl_l3996_399630


namespace coin_distribution_rotations_l3996_399650

/-- Represents the coin distribution problem on a round table. -/
structure CoinDistribution where
  n : ℕ  -- number of sectors and players
  m : ℕ  -- number of rotations
  h_n_ge_4 : n ≥ 4

  /-- Player 1 received 74 fewer coins than player 4 -/
  h_player1_4 : ∃ (c1 c4 : ℕ), c4 - c1 = 74

  /-- Player 2 received 50 fewer coins than player 3 -/
  h_player2_3 : ∃ (c2 c3 : ℕ), c3 - c2 = 50

  /-- Player 4 received 3 coins twice as often as 2 coins -/
  h_player4_3_2 : ∃ (t2 t3 : ℕ), t3 = 2 * t2

  /-- Player 4 received 3 coins half as often as 1 coin -/
  h_player4_3_1 : ∃ (t1 t3 : ℕ), t3 = t1 / 2

/-- The number of rotations in the coin distribution problem is 69. -/
theorem coin_distribution_rotations (cd : CoinDistribution) : cd.m = 69 := by
  sorry

end coin_distribution_rotations_l3996_399650


namespace f_satisfies_conditions_l3996_399699

-- Define the polynomial f
def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

-- Theorem stating that f satisfies all conditions
theorem f_satisfies_conditions :
  -- f is a polynomial in x, y, z (implied by its definition)
  -- f is of degree 4 in x (implied by its definition)
  -- First condition
  (∀ x y z : ℝ, f x (z^2) y + f x (y^2) z = 0) ∧
  -- Second condition
  (∀ x y z : ℝ, f (z^3) y x + f (x^3) y z = 0) :=
by sorry


end f_satisfies_conditions_l3996_399699


namespace nice_people_count_l3996_399603

/-- Represents the proportion of nice people for each name --/
def nice_proportion (name : String) : ℚ :=
  match name with
  | "Barry" => 1
  | "Kevin" => 1/2
  | "Julie" => 3/4
  | "Joe" => 1/10
  | _ => 0

/-- Represents the number of people with each name in the crowd --/
def crowd_count (name : String) : ℕ :=
  match name with
  | "Barry" => 24
  | "Kevin" => 20
  | "Julie" => 80
  | "Joe" => 50
  | _ => 0

/-- Calculates the number of nice people for a given name --/
def nice_count (name : String) : ℕ :=
  (nice_proportion name * crowd_count name).num.toNat

/-- The total number of nice people in the crowd --/
def total_nice_people : ℕ :=
  nice_count "Barry" + nice_count "Kevin" + nice_count "Julie" + nice_count "Joe"

/-- Theorem stating that the total number of nice people in the crowd is 99 --/
theorem nice_people_count : total_nice_people = 99 := by
  sorry

end nice_people_count_l3996_399603


namespace factorization_equality_l3996_399619

theorem factorization_equality (x y : ℝ) : 
  3 * x^3 - 6 * x^2 * y + 3 * x * y^2 = 3 * x * (x - y)^2 := by
  sorry

end factorization_equality_l3996_399619


namespace tan_135_degrees_l3996_399662

/-- Tangent of 135 degrees is -1 -/
theorem tan_135_degrees : Real.tan (135 * π / 180) = -1 := by
  sorry

end tan_135_degrees_l3996_399662


namespace probability_at_least_one_correct_l3996_399601

/-- The probability of getting at least one question right when randomly guessing 5 questions,
    each with 6 answer choices. -/
theorem probability_at_least_one_correct (n : ℕ) (choices : ℕ) : 
  n = 5 → choices = 6 → (1 - (choices - 1 : ℚ) / choices ^ n) = 4651 / 7776 := by
  sorry

#check probability_at_least_one_correct

end probability_at_least_one_correct_l3996_399601


namespace symmetry_center_of_f_l3996_399673

/-- Given a function f(x) and a constant θ, prove that (0,0) is one of the symmetry centers of the graph of f(x). -/
theorem symmetry_center_of_f (θ : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * Real.cos (2 * x + θ) * Real.sin θ - Real.sin (2 * (x + θ))
  (0, 0) ∈ {p : ℝ × ℝ | ∀ x, f (p.1 + x) = f (p.1 - x)} :=
by sorry

end symmetry_center_of_f_l3996_399673


namespace smallest_value_of_floor_sum_l3996_399636

theorem smallest_value_of_floor_sum (a b c : ℕ+) 
  (hab : (a : ℚ) / b = 2)
  (hbc : (b : ℚ) / c = 2)
  (hca : (c : ℚ) / a = 1 / 4) :
  ⌊(a + b : ℚ) / c⌋ + ⌊(b + c : ℚ) / a⌋ + ⌊(c + a : ℚ) / b⌋ = 8 :=
by sorry

end smallest_value_of_floor_sum_l3996_399636


namespace simplify_expression_l3996_399655

theorem simplify_expression (x y : ℝ) : 3*x^2 - 3*(2*x^2 + 4*y) + 2*(x^2 - y) = -x^2 - 14*y := by
  sorry

end simplify_expression_l3996_399655


namespace smallest_number_l3996_399600

def numbers : Finset ℚ := {5, -1/3, 0, -2}

theorem smallest_number : 
  ∀ x ∈ numbers, -2 ≤ x :=
by sorry

end smallest_number_l3996_399600


namespace camille_bird_count_l3996_399663

/-- The number of birds Camille saw while bird watching -/
def total_birds (cardinals robins blue_jays sparrows : ℕ) : ℕ :=
  cardinals + robins + blue_jays + sparrows

/-- Theorem stating the total number of birds Camille saw -/
theorem camille_bird_count :
  ∃ (cardinals robins blue_jays sparrows : ℕ),
    cardinals = 3 ∧
    robins = 4 * cardinals ∧
    blue_jays = 2 * cardinals ∧
    sparrows = 3 * cardinals + 1 ∧
    total_birds cardinals robins blue_jays sparrows = 31 :=
by
  sorry

end camille_bird_count_l3996_399663


namespace BC_length_l3996_399690

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
axiom right_triangle_ABC : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
axiom right_triangle_ABD : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0
axiom AD_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 45
axiom BD_length : Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 52

-- Theorem statement
theorem BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 1079 := by
  sorry

end BC_length_l3996_399690


namespace circumscribed_sphere_surface_area_l3996_399635

theorem circumscribed_sphere_surface_area (cube_volume : ℝ) (h : cube_volume = 27) :
  let cube_side := cube_volume ^ (1/3)
  let sphere_diameter := cube_side * Real.sqrt 3
  let sphere_radius := sphere_diameter / 2
  4 * Real.pi * sphere_radius ^ 2 = 27 * Real.pi :=
by
  sorry

end circumscribed_sphere_surface_area_l3996_399635


namespace selection_from_three_female_two_male_l3996_399611

/-- The number of ways to select one person from a group of female and male students. -/
def selection_methods (num_female : ℕ) (num_male : ℕ) : ℕ :=
  num_female + num_male

/-- Theorem: The number of ways to select one person from 3 female students and 2 male students is 5. -/
theorem selection_from_three_female_two_male :
  selection_methods 3 2 = 5 := by
  sorry

end selection_from_three_female_two_male_l3996_399611


namespace sum4_equivalence_l3996_399679

-- Define the type for a die
def Die := Fin 6

-- Define the sum of two dice
def diceSum (d1 d2 : Die) : Nat := d1.val + d2.val + 2

-- Define the event where the sum is 4
def sumIs4 (d1 d2 : Die) : Prop := diceSum d1 d2 = 4

-- Define the event where one die is 3 and the other is 1
def oneThreeOneOne (d1 d2 : Die) : Prop :=
  (d1.val = 2 ∧ d2.val = 0) ∨ (d1.val = 0 ∧ d2.val = 2)

-- Define the event where both dice show 2
def bothTwo (d1 d2 : Die) : Prop := d1.val = 1 ∧ d2.val = 1

-- Theorem stating the equivalence
theorem sum4_equivalence (d1 d2 : Die) :
  sumIs4 d1 d2 ↔ oneThreeOneOne d1 d2 ∨ bothTwo d1 d2 := by
  sorry


end sum4_equivalence_l3996_399679


namespace fourth_rectangle_area_l3996_399625

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem fourth_rectangle_area 
  (large : Rectangle) 
  (small1 small2 small3 small4 : Rectangle) 
  (h1 : small1.length + small3.length = large.length)
  (h2 : small1.width + small2.width = large.width)
  (h3 : small1.length = small2.length)
  (h4 : small1.width = small3.width)
  (h5 : area large = area small1 + area small2 + area small3 + area small4) :
  area small4 = small2.width * small3.length := by
  sorry

#check fourth_rectangle_area

end fourth_rectangle_area_l3996_399625


namespace soda_difference_l3996_399674

theorem soda_difference (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 79) (h2 : diet_soda = 53) : 
  regular_soda - diet_soda = 26 := by
  sorry

end soda_difference_l3996_399674


namespace quadratic_equation_solution_l3996_399608

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 5 * x + 2
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry


end quadratic_equation_solution_l3996_399608


namespace min_value_of_z_l3996_399610

theorem min_value_of_z (x y : ℝ) :
  x^2 + 3*y^2 + 8*x - 6*y + 30 ≥ 11 ∧
  ∃ (x y : ℝ), x^2 + 3*y^2 + 8*x - 6*y + 30 = 11 := by
sorry

end min_value_of_z_l3996_399610


namespace hypotenuse_increase_bound_l3996_399666

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let c := Real.sqrt (x^2 + y^2)
  let c' := Real.sqrt ((x + 1)^2 + (y + 1)^2)
  c' - c ≤ Real.sqrt 2 := by
sorry

end hypotenuse_increase_bound_l3996_399666


namespace fly_distance_l3996_399646

/-- The distance traveled by a fly between two runners --/
theorem fly_distance (joe_speed maria_speed fly_speed initial_distance : ℝ) :
  joe_speed = 10 ∧ 
  maria_speed = 8 ∧ 
  fly_speed = 15 ∧ 
  initial_distance = 3 →
  (fly_speed * initial_distance) / (joe_speed + maria_speed) = 5/2 := by
  sorry

#check fly_distance

end fly_distance_l3996_399646


namespace floor_e_equals_two_l3996_399618

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_equals_two_l3996_399618


namespace line_direction_vector_l3996_399659

/-- Given a line passing through two points and its direction vector, prove the value of 'a' -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (a : ℝ) : 
  p1 = (-3, 7) → p2 = (2, -1) → ∃ k : ℝ, k ≠ 0 ∧ k • (p2.1 - p1.1, p2.2 - p1.2) = (a, -2) → a = 5/4 := by
  sorry

end line_direction_vector_l3996_399659


namespace max_value_of_g_l3996_399629

def g (x : ℝ) : ℝ := 4*x - x^4

theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 2 ∧
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 2 → g x ≤ g c) ∧
  g c = 3 := by
sorry

end max_value_of_g_l3996_399629


namespace least_positive_t_for_geometric_progression_l3996_399649

open Real

theorem least_positive_t_for_geometric_progression (α : ℝ) (h : 0 < α ∧ α < π / 2) :
  ∃ t : ℝ, t > 0 ∧
  (∀ r : ℝ, r > 0 →
    (arcsin (sin α) = r * α ∧
     arcsin (sin (3 * α)) = r^2 * α ∧
     arcsin (sin (5 * α)) = r^3 * α ∧
     arcsin (sin (t * α)) = r^4 * α)) ∧
  (∀ s : ℝ, s > 0 →
    (∃ r : ℝ, r > 0 ∧
      arcsin (sin α) = r * α ∧
      arcsin (sin (3 * α)) = r^2 * α ∧
      arcsin (sin (5 * α)) = r^3 * α ∧
      arcsin (sin (s * α)) = r^4 * α) →
    t ≤ s) ∧
  t = 3 * (π - 5 * α) / (π - 3 * α) :=
by sorry

end least_positive_t_for_geometric_progression_l3996_399649


namespace star_equation_two_roots_l3996_399613

/-- Custom binary operation on real numbers -/
def star (a b : ℝ) : ℝ := a * b^2 - b

/-- Theorem stating the condition for the equation 1※x = k to have two distinct real roots -/
theorem star_equation_two_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ star 1 x₁ = k ∧ star 1 x₂ = k) ↔ k > -1/4 :=
sorry

end star_equation_two_roots_l3996_399613


namespace sin_minus_cos_eq_one_solution_set_l3996_399602

theorem sin_minus_cos_eq_one_solution_set :
  {x : ℝ | Real.sin (x / 2) - Real.cos (x / 2) = 1} =
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.pi / 4 ∨ x = k * Real.pi + Real.pi / 2} := by
  sorry

end sin_minus_cos_eq_one_solution_set_l3996_399602


namespace difference_of_squares_24_13_l3996_399641

theorem difference_of_squares_24_13 : (24 + 13)^2 - (24 - 13)^2 = 407 := by
  sorry

end difference_of_squares_24_13_l3996_399641


namespace parabola_focus_distance_l3996_399667

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola in the first quadrant
def point_on_parabola (Q : ℝ × ℝ) : Prop :=
  parabola Q.1 Q.2 ∧ Q.1 > 0 ∧ Q.2 > 0

-- Define the condition for vector PQ and QF
def vector_condition (P Q : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 2 * ((Q.1 - focus.1)^2 + (Q.2 - focus.2)^2)

-- Main theorem
theorem parabola_focus_distance 
  (Q : ℝ × ℝ) 
  (h1 : point_on_parabola Q) 
  (h2 : ∃ P, vector_condition P Q) : 
  (Q.1 - focus.1)^2 + (Q.2 - focus.2)^2 = (8 + 4*Real.sqrt 2)^2 :=
sorry

end parabola_focus_distance_l3996_399667


namespace multiplier_value_l3996_399658

theorem multiplier_value (n : ℕ) (increase : ℕ) (result : ℕ) : 
  n = 14 → increase = 196 → result = 15 → n * result = n + increase :=
by
  sorry

end multiplier_value_l3996_399658


namespace orange_distribution_theorem_l3996_399684

/-- Represents the number of oranges each person has at a given stage -/
structure OrangeDistribution :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Defines the redistribution rules for oranges -/
def redistribute (d : OrangeDistribution) : OrangeDistribution :=
  let d1 := OrangeDistribution.mk (d.a / 2) (d.b + d.a / 2) d.c
  let d2 := OrangeDistribution.mk d1.a (d1.b * 4 / 5) (d1.c + d1.b / 5)
  OrangeDistribution.mk (d2.a + d2.c / 7) d2.b (d2.c * 6 / 7)

theorem orange_distribution_theorem (initial : OrangeDistribution) :
  initial.a + initial.b + initial.c = 108 →
  let final := redistribute initial
  final.a = final.b ∧ final.b = final.c →
  initial = OrangeDistribution.mk 72 9 27 := by
  sorry

end orange_distribution_theorem_l3996_399684


namespace friends_initial_money_l3996_399626

theorem friends_initial_money (your_initial_money : ℕ) (your_weekly_savings : ℕ) 
  (friend_weekly_savings : ℕ) (weeks : ℕ) :
  your_initial_money = 160 →
  your_weekly_savings = 7 →
  friend_weekly_savings = 5 →
  weeks = 25 →
  ∃ (friend_initial_money : ℕ),
    your_initial_money + your_weekly_savings * weeks = 
    friend_initial_money + friend_weekly_savings * weeks ∧
    friend_initial_money = 210 :=
by sorry

end friends_initial_money_l3996_399626


namespace negative_integers_satisfying_condition_l3996_399698

def satisfies_condition (a : Int) : Prop :=
  (4 * a + 1 : ℚ) / 6 > -2

theorem negative_integers_satisfying_condition :
  {a : Int | a < 0 ∧ satisfies_condition a} = {-1, -2, -3} := by
  sorry

end negative_integers_satisfying_condition_l3996_399698


namespace heart_properties_l3996_399637

def heart (x y : ℝ) : ℝ := |x - y|

theorem heart_properties :
  ∀ x y : ℝ,
  (heart x y = heart y x) ∧
  (3 * heart x y = heart (3 * x) (3 * y)) ∧
  (heart (x + 1) (y + 1) = heart x y) ∧
  (heart x x = 0) ∧
  (heart x y ≥ 0) ∧
  (heart x y > 0 ↔ x ≠ y) := by
  sorry

end heart_properties_l3996_399637


namespace product_equality_l3996_399696

theorem product_equality : 500 * 2468 * 0.2468 * 100 = 30485120 := by
  sorry

end product_equality_l3996_399696


namespace sum_of_possible_S_values_l3996_399670

theorem sum_of_possible_S_values : ∃ (a b c x y z : ℕ+) (S : ℕ),
  (a^2 - 2 : ℚ) / x = (b^2 - 37 : ℚ) / y ∧
  (b^2 - 37 : ℚ) / y = (c^2 - 41 : ℚ) / z ∧
  (c^2 - 41 : ℚ) / z = (a + b + c : ℚ) ∧
  S = a + b + c + x + y + z ∧
  (∀ (a' b' c' x' y' z' : ℕ+) (S' : ℕ),
    ((a'^2 - 2 : ℚ) / x' = (b'^2 - 37 : ℚ) / y' ∧
     (b'^2 - 37 : ℚ) / y' = (c'^2 - 41 : ℚ) / z' ∧
     (c'^2 - 41 : ℚ) / z' = (a' + b' + c' : ℚ) ∧
     S' = a' + b' + c' + x' + y' + z') →
    S = 98 ∨ S = 211) ∧
  (∃ (a₁ b₁ c₁ x₁ y₁ z₁ : ℕ+) (S₁ : ℕ),
    (a₁^2 - 2 : ℚ) / x₁ = (b₁^2 - 37 : ℚ) / y₁ ∧
    (b₁^2 - 37 : ℚ) / y₁ = (c₁^2 - 41 : ℚ) / z₁ ∧
    (c₁^2 - 41 : ℚ) / z₁ = (a₁ + b₁ + c₁ : ℚ) ∧
    S₁ = a₁ + b₁ + c₁ + x₁ + y₁ + z₁ ∧
    S₁ = 98) ∧
  (∃ (a₂ b₂ c₂ x₂ y₂ z₂ : ℕ+) (S₂ : ℕ),
    (a₂^2 - 2 : ℚ) / x₂ = (b₂^2 - 37 : ℚ) / y₂ ∧
    (b₂^2 - 37 : ℚ) / y₂ = (c₂^2 - 41 : ℚ) / z₂ ∧
    (c₂^2 - 41 : ℚ) / z₂ = (a₂ + b₂ + c₂ : ℚ) ∧
    S₂ = a₂ + b₂ + c₂ + x₂ + y₂ + z₂ ∧
    S₂ = 211) :=
by
  sorry

end sum_of_possible_S_values_l3996_399670


namespace perpendicular_lines_m_eq_6_l3996_399683

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (a b c d e f : ℝ), a ≠ 0 ∧ d ≠ 0 ∧ 
    (∀ x y, a*x + b*y + c = 0 ↔ y = m₁*x + (c/b)) ∧
    (∀ x y, d*x + e*y + f = 0 ↔ y = m₂*x + (f/e)))

/-- The theorem to be proved -/
theorem perpendicular_lines_m_eq_6 :
  ∀ (m : ℝ), (∀ x y, x - 2*y - 3 = 0 ↔ y = (1/2)*x - 3/2) →
             (∀ x y, m*x + 3*y - 6 = 0 ↔ y = (-m/3)*x + 2) →
             (∃ (m₁ m₂ : ℝ), m₁ * m₂ = -1 ∧ 
               (∀ x y, x - 2*y - 3 = 0 ↔ y = m₁*x - 3/2) ∧
               (∀ x y, m*x + 3*y - 6 = 0 ↔ y = m₂*x + 2)) →
             m = 6 := by
  sorry

end perpendicular_lines_m_eq_6_l3996_399683


namespace intersection_point_determines_intercept_l3996_399616

/-- Given two lines that intersect at the same point as a third line, find the y-intercept of the third line -/
theorem intersection_point_determines_intercept 
  (line1 : ℝ → ℝ) (line2 : ℝ → ℝ) (line3 : ℝ → ℝ → ℝ) 
  (h1 : ∀ x, line1 x = 3 * x + 7)
  (h2 : ∀ x, line2 x = -4 * x + 1)
  (h3 : ∀ x k, line3 x k = 2 * x + k)
  (h_intersect : ∃ x y k, line1 x = y ∧ line2 x = y ∧ line3 x k = y) :
  ∃ k, k = 43 / 7 ∧ ∀ x, line3 x k = line1 x ∧ line3 x k = line2 x := by
  sorry

end intersection_point_determines_intercept_l3996_399616


namespace star_property_l3996_399638

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.one
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.four
  | Element.two, Element.one => Element.three
  | Element.two, Element.two => Element.one
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.one

theorem star_property :
  star (star Element.three Element.two) (star Element.four Element.one) = Element.one := by
  sorry

end star_property_l3996_399638


namespace problem_1_problem_2_l3996_399694

-- Problem 1
theorem problem_1 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + 2*y)^2 - (-2*x*y^2)^2 / (x*y^3) = x^2 + 4*y^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx1 : x ≠ 1) (hx3 : x ≠ 3) :
  (x - 1) / (x - 3) * (2 - x + 2 / (x - 1)) = -x := by sorry

end problem_1_problem_2_l3996_399694


namespace min_abs_z_plus_one_l3996_399604

theorem min_abs_z_plus_one (z : ℂ) (h : Complex.abs (z^2 + 1) = Complex.abs (z * (z + Complex.I))) :
  ∃ (w : ℂ), ∀ (z : ℂ), Complex.abs (z^2 + 1) = Complex.abs (z * (z + Complex.I)) →
    Complex.abs (w + 1) ≤ Complex.abs (z + 1) ∧ Complex.abs (w + 1) = 0 :=
by sorry

end min_abs_z_plus_one_l3996_399604


namespace population_growth_percentage_l3996_399660

theorem population_growth_percentage (a b c : ℝ) : 
  let growth_factor_1 := 1 + a / 100
  let growth_factor_2 := 1 + b / 100
  let growth_factor_3 := 1 + c / 100
  let total_growth := growth_factor_1 * growth_factor_2 * growth_factor_3
  (total_growth - 1) * 100 = a + b + c + (a * b + a * c + b * c) / 100 + a * b * c / 10000 := by
sorry

end population_growth_percentage_l3996_399660


namespace max_sin_product_right_triangle_l3996_399675

/-- For any right triangle ABC with angle C = 90°, the maximum value of sin A sin B is 1/2. -/
theorem max_sin_product_right_triangle (A B C : Real) : 
  0 ≤ A ∧ 0 ≤ B ∧ -- Angles are non-negative
  A + B + C = π ∧ -- Sum of angles in a triangle is π
  C = π / 2 → -- Right angle at C
  ∀ (x y : Real), 0 ≤ x ∧ 0 ≤ y ∧ x + y + π/2 = π → 
    Real.sin A * Real.sin B ≤ Real.sin x * Real.sin y ∧
    Real.sin A * Real.sin B ≤ (1 : Real) / 2 := by
  sorry

end max_sin_product_right_triangle_l3996_399675


namespace two_solutions_cubic_equation_l3996_399633

theorem two_solutions_cubic_equation : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^3 + y^2 = 2*y + 1) ∧ 
    s.card = 2 := by
  sorry

end two_solutions_cubic_equation_l3996_399633


namespace cindy_same_color_prob_l3996_399652

/-- Represents the number of marbles of each color in the box -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles -/
def totalMarbles (mc : MarbleCount) : ℕ := mc.red + mc.green + mc.yellow

/-- Represents the number of marbles drawn by each person -/
structure DrawCounts where
  alice : ℕ
  bob : ℕ
  cindy : ℕ

/-- Calculates the probability of Cindy getting 3 marbles of the same color -/
noncomputable def probCindySameColor (mc : MarbleCount) (dc : DrawCounts) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem cindy_same_color_prob :
  let initial_marbles : MarbleCount := ⟨2, 2, 4⟩
  let draw_counts : DrawCounts := ⟨2, 3, 3⟩
  probCindySameColor initial_marbles draw_counts = 13 / 140 :=
sorry

end cindy_same_color_prob_l3996_399652


namespace tan_sum_reciprocal_l3996_399657

theorem tan_sum_reciprocal (x y : ℝ) 
  (h1 : Real.sin x / Real.cos y + Real.sin y / Real.cos x = 2)
  (h2 : Real.cos x / Real.sin y + Real.cos y / Real.sin x = 4) :
  Real.tan x / Real.tan y + Real.tan y / Real.tan x = 4 := by
  sorry

end tan_sum_reciprocal_l3996_399657


namespace economic_output_equals_scientific_notation_l3996_399686

/-- Represents the economic output in yuan -/
def economic_output : ℝ := 4500 * 1000000000

/-- The scientific notation representation of the economic output -/
def scientific_notation : ℝ := 4.5 * (10 ^ 12)

/-- Theorem stating that the economic output is equal to its scientific notation representation -/
theorem economic_output_equals_scientific_notation : 
  economic_output = scientific_notation := by sorry

end economic_output_equals_scientific_notation_l3996_399686


namespace fault_line_movement_l3996_399605

/-- The total movement of a fault line over two years, given its movement in each year -/
def total_movement (movement_year1 : ℝ) (movement_year2 : ℝ) : ℝ :=
  movement_year1 + movement_year2

/-- Theorem stating that the total movement of the fault line over two years is 6.50 inches -/
theorem fault_line_movement :
  let movement_year1 : ℝ := 1.25
  let movement_year2 : ℝ := 5.25
  total_movement movement_year1 movement_year2 = 6.50 := by
  sorry

end fault_line_movement_l3996_399605


namespace sqrt_cube_root_power_six_l3996_399692

theorem sqrt_cube_root_power_six : (Real.sqrt ((Real.sqrt 3) ^ 4)) ^ 6 = 729 := by
  sorry

end sqrt_cube_root_power_six_l3996_399692


namespace max_value_ab_squared_l3996_399628

theorem max_value_ab_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∃ (max : ℝ), max = (4 * Real.sqrt 6) / 9 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 2 → x * y^2 ≤ max :=
by sorry

end max_value_ab_squared_l3996_399628


namespace ryan_quiz_goal_l3996_399607

theorem ryan_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ)
  (mid_year_quizzes : ℕ) (mid_year_as : ℕ) (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 3/4) (h3 : mid_year_quizzes = 40) (h4 : mid_year_as = 30) :
  ∃ (max_lower_grade : ℕ),
    max_lower_grade = 5 ∧
    (mid_year_as + (total_quizzes - mid_year_quizzes - max_lower_grade) : ℚ) / total_quizzes ≥ goal_percentage :=
by sorry

end ryan_quiz_goal_l3996_399607


namespace function_property_l3996_399695

def IsMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem function_property (f : ℝ → ℝ) (h_monotonic : IsMonotonic f) 
    (h_property : ∀ x > 0, f (f x + 2 / x) = 1) : 
  f 1 = 0 := by
  sorry

end function_property_l3996_399695


namespace probability_of_middle_position_l3996_399668

theorem probability_of_middle_position (n : ℕ) (h : n = 3) :
  (2 : ℚ) / (n.factorial : ℚ) = (1 : ℚ) / 3 :=
by sorry

end probability_of_middle_position_l3996_399668


namespace min_value_function_l3996_399661

theorem min_value_function (x : ℝ) (hx : x > 1) : 
  let m := (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2)
  let y := 2 * m * x + 3 / (x - 1) + 1
  y ≥ 2 + 2 * Real.sqrt 3 := by
sorry

end min_value_function_l3996_399661


namespace geometric_arithmetic_sequence_ratio_l3996_399672

/-- Given a positive geometric sequence {a_n} where a_2, a_3/2, a_1 form an arithmetic sequence,
    prove that (a_4 + a_5) / (a_3 + a_4) = (1 + √5) / 2 -/
theorem geometric_arithmetic_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : ∃ d : ℝ, a 2 - a 3 / 2 = a 3 / 2 - a 1) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 := by
  sorry

end geometric_arithmetic_sequence_ratio_l3996_399672


namespace fraction_calculation_l3996_399640

theorem fraction_calculation : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end fraction_calculation_l3996_399640


namespace cookie_sheet_width_l3996_399681

/-- Represents a rectangle with given width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.length)

/-- Theorem: A rectangle with length 2 and perimeter 24 has width 10 -/
theorem cookie_sheet_width : 
  ∀ (r : Rectangle), r.length = 2 → r.perimeter = 24 → r.width = 10 := by
  sorry

end cookie_sheet_width_l3996_399681


namespace bottles_per_case_l3996_399676

/-- The number of bottles a case can hold, given the total daily production and number of cases required. -/
theorem bottles_per_case (total_bottles : ℕ) (total_cases : ℕ) 
  (h1 : total_bottles = 72000) 
  (h2 : total_cases = 8000) : 
  total_bottles / total_cases = 9 := by
sorry

end bottles_per_case_l3996_399676


namespace gym_distance_difference_l3996_399647

/-- The distance from Anthony's apartment to work in miles -/
def distance_to_work : ℝ := 10

/-- The distance from Anthony's apartment to the gym in miles -/
def distance_to_gym : ℝ := 7

/-- The distance to the gym is more than half the distance to work -/
axiom gym_further_than_half : distance_to_gym > distance_to_work / 2

theorem gym_distance_difference : distance_to_gym - distance_to_work / 2 = 2 := by
  sorry

end gym_distance_difference_l3996_399647


namespace purely_imaginary_complex_number_l3996_399614

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := Complex.mk (a - 1) 1
  (∀ x : ℝ, z = Complex.I * x) → a = 1 := by
  sorry

end purely_imaginary_complex_number_l3996_399614


namespace max_imaginary_part_of_roots_l3996_399697

open Complex

theorem max_imaginary_part_of_roots (z : ℂ) (k : ℤ) :
  z^12 - z^9 + z^6 - z^3 + 1 = 0 →
  z = exp (I * Real.pi * (1/15 + 2/15 * k)) →
  ∃ θ : ℝ, -Real.pi/2 ≤ θ ∧ θ ≤ Real.pi/2 ∧
    (∀ w : ℂ, w^12 - w^9 + w^6 - w^3 + 1 = 0 →
      Complex.abs (Complex.im w) ≤ Real.sin (7*Real.pi/30)) :=
by sorry

end max_imaginary_part_of_roots_l3996_399697


namespace arctg_sum_implies_product_sum_l3996_399606

/-- Given that arctg x + arctg y + arctg z = π/2, prove that xy + yz + zx = 1 -/
theorem arctg_sum_implies_product_sum (x y z : ℝ) 
  (h : Real.arctan x + Real.arctan y + Real.arctan z = π / 2) : 
  x * y + y * z + x * z = 1 := by
  sorry

end arctg_sum_implies_product_sum_l3996_399606


namespace circle_chords_with_equal_sums_l3996_399651

/-- Given 2^500 points on a circle labeled 1 to 2^500, there exist 100 pairwise disjoint chords
    such that the sums of the labels at their endpoints are all equal. -/
theorem circle_chords_with_equal_sums :
  ∀ (labeling : Fin (2^500) → Fin (2^500)),
  ∃ (chords : Finset (Fin (2^500) × Fin (2^500))),
    (chords.card = 100) ∧
    (∀ (c1 c2 : Fin (2^500) × Fin (2^500)), c1 ∈ chords → c2 ∈ chords → c1 ≠ c2 → 
      (c1.1 ≠ c2.1 ∧ c1.1 ≠ c2.2 ∧ c1.2 ≠ c2.1 ∧ c1.2 ≠ c2.2)) ∧
    (∃ (sum : Nat), ∀ (c : Fin (2^500) × Fin (2^500)), c ∈ chords → 
      (labeling c.1).val + (labeling c.2).val = sum) :=
by sorry

end circle_chords_with_equal_sums_l3996_399651


namespace bus_speed_excluding_stoppages_l3996_399634

/-- Proves that given a bus with a speed of 42 kmph including stoppages
    and stopping for 9.6 minutes per hour, the speed excluding stoppages is 50 kmph. -/
theorem bus_speed_excluding_stoppages
  (speed_with_stoppages : ℝ)
  (stoppage_time : ℝ)
  (h1 : speed_with_stoppages = 42)
  (h2 : stoppage_time = 9.6)
  : (speed_with_stoppages * 60) / (60 - stoppage_time) = 50 := by
  sorry

end bus_speed_excluding_stoppages_l3996_399634


namespace largest_four_digit_divisible_by_sum_of_digits_l3996_399689

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_divisible_by_sum_of_digits :
  ∃ (n : ℕ), is_four_digit n ∧ n % (sum_of_digits n) = 0 ∧
  ∀ (m : ℕ), is_four_digit m ∧ m % (sum_of_digits m) = 0 → m ≤ n :=
by
  use 9990
  sorry

end largest_four_digit_divisible_by_sum_of_digits_l3996_399689


namespace square_fraction_l3996_399654

theorem square_fraction (a b : ℕ+) (h : (a.val * b.val + 1) ∣ (a.val^2 + b.val^2)) :
  ∃ (k : ℕ), (a.val^2 + b.val^2) / (a.val * b.val + 1) = k^2 := by
  sorry

end square_fraction_l3996_399654


namespace negative_two_to_fourth_power_l3996_399639

theorem negative_two_to_fourth_power : -2 * 2 * 2 * 2 = -2^4 := by
  sorry

end negative_two_to_fourth_power_l3996_399639


namespace system_solution_l3996_399620

-- Define the system of equations
def system_equations (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (n ≥ 3) ∧
  (∀ i ∈ Finset.range (n - 1), x i ^ 3 = x ((i + 1) % n) + x ((i + 2) % n) + 1) ∧
  (x (n - 1) ^ 3 = x 0 + x 1 + 1)

-- Define the solution set
def solution_set : Set ℝ :=
  {-1, (1 + Real.sqrt 5) / 2, (1 - Real.sqrt 5) / 2}

-- Theorem statement
theorem system_solution (n : ℕ) (x : ℕ → ℝ) :
  system_equations n x →
  (∀ i ∈ Finset.range n, x i ∈ solution_set) ∧
  (∃ t ∈ solution_set, ∀ i ∈ Finset.range n, x i = t) :=
by sorry

end system_solution_l3996_399620


namespace lisa_marble_distribution_l3996_399682

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisa_marble_distribution (num_friends : ℕ) (initial_marbles : ℕ)
  (h1 : num_friends = 14)
  (h2 : initial_marbles = 50) :
  min_additional_marbles num_friends initial_marbles = 55 := by
  sorry

end lisa_marble_distribution_l3996_399682


namespace average_age_of_nine_students_l3996_399624

theorem average_age_of_nine_students 
  (total_students : ℕ) 
  (average_age_all : ℝ) 
  (students_group1 : ℕ) 
  (average_age_group1 : ℝ) 
  (age_last_student : ℕ) 
  (h1 : total_students = 20)
  (h2 : average_age_all = 20)
  (h3 : students_group1 = 10)
  (h4 : average_age_group1 = 24)
  (h5 : age_last_student = 61) :
  let students_group2 := total_students - students_group1 - 1
  let total_age_all := average_age_all * total_students
  let total_age_group1 := average_age_group1 * students_group1
  let total_age_group2 := total_age_all - total_age_group1 - age_last_student
  (total_age_group2 / students_group2 : ℝ) = 11 := by
  sorry

end average_age_of_nine_students_l3996_399624


namespace sheila_weekly_earnings_l3996_399617

/-- Represents Sheila's work schedule and earnings --/
structure SheilaWork where
  hourly_rate : ℕ
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  days_mon_wed_fri : ℕ
  days_tue_thu : ℕ

/-- Calculates Sheila's weekly earnings --/
def weekly_earnings (s : SheilaWork) : ℕ :=
  s.hourly_rate * (s.hours_mon_wed_fri * s.days_mon_wed_fri + s.hours_tue_thu * s.days_tue_thu)

/-- Theorem stating Sheila's weekly earnings --/
theorem sheila_weekly_earnings :
  ∃ (s : SheilaWork),
    s.hourly_rate = 13 ∧
    s.hours_mon_wed_fri = 8 ∧
    s.hours_tue_thu = 6 ∧
    s.days_mon_wed_fri = 3 ∧
    s.days_tue_thu = 2 ∧
    weekly_earnings s = 468 := by
  sorry

end sheila_weekly_earnings_l3996_399617


namespace prob_red_black_correct_l3996_399687

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (h_total : total_cards = 52)
  (h_red : red_cards = 26)
  (h_black : black_cards = 26)
  (h_sum : red_cards + black_cards = total_cards)

/-- The probability of drawing one red card and one black card in the first two draws -/
def prob_red_black (d : Deck) : ℚ :=
  26 / 51

theorem prob_red_black_correct (d : Deck) : 
  prob_red_black d = 26 / 51 := by
  sorry

end prob_red_black_correct_l3996_399687


namespace smallest_n_for_divisible_sum_of_squares_l3996_399609

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem smallest_n_for_divisible_sum_of_squares :
  ∀ n : ℕ, n > 0 → (sum_of_squares n % 100 = 0 → n ≥ 24) ∧
  (sum_of_squares 24 % 100 = 0) :=
sorry

end smallest_n_for_divisible_sum_of_squares_l3996_399609


namespace certain_number_exists_l3996_399693

theorem certain_number_exists : ∃ x : ℝ, ((x + 10) * 2 / 2)^3 - 2 = 120 / 3 := by
  sorry

end certain_number_exists_l3996_399693


namespace function_properties_l3996_399680

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f x * f y = (f (x + y) + 2 * f (x - y)) / 3)
  (h2 : ∀ x : ℝ, f x ≠ 0) :
  (f 0 = 1) ∧ (∀ x : ℝ, f x = f (-x)) := by sorry

end function_properties_l3996_399680


namespace possible_m_values_l3996_399691

def A : Set ℝ := {x | x^2 - 9*x - 10 = 0}

def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem possible_m_values :
  ∀ m : ℝ, (A ∪ B m = A) ↔ m ∈ ({0, 1, -(1/10)} : Set ℝ) := by sorry

end possible_m_values_l3996_399691


namespace percentage_of_girls_in_class_l3996_399622

theorem percentage_of_girls_in_class (B G : ℝ) :
  B > 0 →
  G > 0 →
  G + 0.5 * B = 1.5 * (0.5 * B) →
  (G / (B + G)) * 100 = 20 := by
sorry

end percentage_of_girls_in_class_l3996_399622
