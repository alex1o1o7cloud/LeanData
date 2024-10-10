import Mathlib

namespace spring_mass_for_27cm_unique_mass_for_27cm_l3285_328504

-- Define the relationship between spring length and mass
def spring_length (mass : ℝ) : ℝ := 16 + 2 * mass

-- Theorem stating that when the spring length is 27 cm, the mass is 5.5 kg
theorem spring_mass_for_27cm : spring_length 5.5 = 27 := by
  sorry

-- Theorem stating the uniqueness of the solution
theorem unique_mass_for_27cm (mass : ℝ) : 
  spring_length mass = 27 → mass = 5.5 := by
  sorry

end spring_mass_for_27cm_unique_mass_for_27cm_l3285_328504


namespace person_peach_count_l3285_328534

theorem person_peach_count (jake_peaches jake_apples person_apples person_peaches : ℕ) : 
  jake_peaches + 6 = person_peaches →
  jake_apples = person_apples + 8 →
  person_apples = 16 →
  person_peaches = person_apples + 1 →
  person_peaches = 17 := by
sorry

end person_peach_count_l3285_328534


namespace two_numbers_with_specific_means_l3285_328552

theorem two_numbers_with_specific_means :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  (a + b) / 2 = 5 ∧
  a = 5 + 2 * Real.sqrt 5 ∧
  b = 5 - 2 * Real.sqrt 5 := by
    sorry

end two_numbers_with_specific_means_l3285_328552


namespace prism_diagonals_l3285_328547

/-- Checks if three numbers can be the lengths of external diagonals of a right regular prism -/
def valid_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 > c^2 ∧
  b^2 + c^2 > a^2 ∧
  a^2 + c^2 > b^2

theorem prism_diagonals :
  ¬(valid_diagonals 6 8 11) ∧
  (valid_diagonals 6 8 10) ∧
  (valid_diagonals 6 10 11) ∧
  (valid_diagonals 8 10 11) ∧
  (valid_diagonals 8 11 12) :=
by sorry

end prism_diagonals_l3285_328547


namespace log_27_3_l3285_328570

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end log_27_3_l3285_328570


namespace jessica_attended_two_games_l3285_328536

/-- The number of soccer games Jessica attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

/-- Proof that Jessica attended 2 games -/
theorem jessica_attended_two_games :
  games_attended 6 4 = 2 := by
  sorry

end jessica_attended_two_games_l3285_328536


namespace min_value_of_b_is_negative_two_l3285_328598

/-- The function that represents b in terms of a, where y = 2x + b is a tangent line to y = a ln x --/
noncomputable def b (a : ℝ) : ℝ := a * Real.log (a / 2) - a

/-- The theorem stating that the minimum value of b is -2 when a > 0 --/
theorem min_value_of_b_is_negative_two :
  ∀ a : ℝ, a > 0 → (∀ x : ℝ, x > 0 → b x ≥ b 2) ∧ b 2 = -2 := by sorry

end min_value_of_b_is_negative_two_l3285_328598


namespace ob_value_l3285_328582

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the points
variable (O F₁ F₂ A B : ℝ × ℝ)

-- State the conditions
variable (h1 : O = (0, 0))
variable (h2 : ellipse (F₁.1) (F₁.2))
variable (h3 : ellipse (F₂.1) (F₂.2))
variable (h4 : F₁.1 < 0 ∧ F₂.1 > 0)
variable (h5 : ellipse A.1 A.2)
variable (h6 : (A.1 - F₂.1) * (F₂.1 - F₁.1) + (A.2 - F₂.2) * (F₂.2 - F₁.2) = 0)
variable (h7 : B.1 = 0)
variable (h8 : ∃ t : ℝ, B = t • F₁ + (1 - t) • A)

-- State the theorem
theorem ob_value : abs B.2 = 3/4 := by sorry

end ob_value_l3285_328582


namespace derivative_of_y_at_2_l3285_328506

-- Define the function y = 3x
def y (x : ℝ) : ℝ := 3 * x

-- State the theorem
theorem derivative_of_y_at_2 :
  deriv y 2 = 3 := by sorry

end derivative_of_y_at_2_l3285_328506


namespace tax_problem_l3285_328503

/-- Proves that the monthly gross income is 127,500 HUF when the tax equals 30% of the annual income --/
theorem tax_problem (x : ℝ) (h1 : x > 1050000) : 
  (267000 + 0.4 * (x - 1050000) = 0.3 * x) → (x / 12 = 127500) := by
  sorry

end tax_problem_l3285_328503


namespace x_value_l3285_328595

theorem x_value (x y : ℚ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end x_value_l3285_328595


namespace bob_arrival_probability_bob_arrival_probability_value_l3285_328548

/-- The probability that Bob arrived before 3:45 PM given that Alice arrived after him,
    when both arrive randomly between 3:00 PM and 4:00 PM. -/
theorem bob_arrival_probability : ℝ :=
  let total_time := 60 -- minutes
  let bob_early_time := 45 -- minutes
  let total_area := (total_time ^ 2) / 2 -- area where Alice arrives after Bob
  let early_area := (bob_early_time ^ 2) / 2 -- area where Bob is early and Alice is after
  early_area / total_area

/-- The probability is equal to 9/16 -/
theorem bob_arrival_probability_value : bob_arrival_probability = 9 / 16 := by
  sorry

end bob_arrival_probability_bob_arrival_probability_value_l3285_328548


namespace sum_of_a_values_for_single_solution_l3285_328516

theorem sum_of_a_values_for_single_solution (a : ℝ) : 
  let equation := fun (x : ℝ) => 2 * x^2 + a * x + 6 * x + 7
  let discriminant := (a + 6)^2 - 4 * 2 * 7
  let sum_of_a_values := -(12 : ℝ)
  (∃ (a₁ a₂ : ℝ), 
    (∀ x, equation x = 0 → discriminant = 0) ∧ 
    (a₁ ≠ a₂) ∧ 
    (a₁ + a₂ = sum_of_a_values)) := by
  sorry

end sum_of_a_values_for_single_solution_l3285_328516


namespace sufficient_not_necessary_l3285_328532

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x < 1) ∧ 
  (∃ y : ℝ, y < 1 ∧ ¬(y > 1)) :=
sorry

end sufficient_not_necessary_l3285_328532


namespace tim_works_six_days_l3285_328508

/-- Represents Tim's work schedule and earnings --/
structure TimsWork where
  tasks_per_day : ℕ
  pay_per_task : ℚ
  weekly_earnings : ℚ

/-- Calculates the number of days Tim works per week --/
def days_worked (w : TimsWork) : ℚ :=
  w.weekly_earnings / (w.tasks_per_day * w.pay_per_task)

/-- Theorem stating that Tim works 6 days a week --/
theorem tim_works_six_days (w : TimsWork) 
  (h1 : w.tasks_per_day = 100)
  (h2 : w.pay_per_task = 6/5) -- $1.2 represented as a fraction
  (h3 : w.weekly_earnings = 720) :
  days_worked w = 6 := by
  sorry

#eval days_worked { tasks_per_day := 100, pay_per_task := 6/5, weekly_earnings := 720 }

end tim_works_six_days_l3285_328508


namespace gift_exchange_equation_l3285_328524

/-- Represents a gathering of people exchanging gifts -/
structure Gathering where
  /-- The number of attendees -/
  attendees : ℕ
  /-- The total number of gifts exchanged -/
  gifts : ℕ
  /-- Each pair of attendees exchanges a different small gift -/
  unique_exchanges : ∀ (a b : Fin attendees), a ≠ b → True

/-- The theorem stating the relationship between attendees and gifts exchanged -/
theorem gift_exchange_equation (g : Gathering) (h : g.gifts = 56) :
  g.attendees * (g.attendees - 1) = 56 := by
  sorry

end gift_exchange_equation_l3285_328524


namespace solution_set_part1_range_of_a_part2_l3285_328587

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem solution_set_part1 :
  ∀ x : ℝ, f (-12) (-2) x < 0 ↔ -1/2 < x ∧ x < 1/3 := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a (-1) x ≥ 0) ↔ a ≥ 1/8 := by sorry

end solution_set_part1_range_of_a_part2_l3285_328587


namespace complement_of_A_union_B_l3285_328551

def U : Set ℕ := {0,1,2,3,4,5}
def A : Set ℕ := {1,2}
def B : Set ℕ := {x ∈ U | x^2 - 5*x + 4 < 0}

theorem complement_of_A_union_B (x : ℕ) : 
  x ∈ (U \ (A ∪ B)) ↔ x ∈ ({0,4,5} : Set ℕ) := by
  sorry

end complement_of_A_union_B_l3285_328551


namespace snooker_tournament_ticket_difference_l3285_328505

theorem snooker_tournament_ticket_difference :
  ∀ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = 320 →
    40 * vip_tickets + 10 * general_tickets = 7500 →
    general_tickets - vip_tickets = 34 := by
  sorry

end snooker_tournament_ticket_difference_l3285_328505


namespace petes_number_l3285_328510

theorem petes_number (x : ℝ) : 3 * (2 * x + 12) = 90 → x = 9 := by
  sorry

end petes_number_l3285_328510


namespace smallest_product_l3285_328539

def digits : List Nat := [5, 6, 7, 8]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat, is_valid_arrangement a b c d →
    product a b c d ≥ 3876 :=
by sorry

end smallest_product_l3285_328539


namespace geometric_sequence_shift_l3285_328586

theorem geometric_sequence_shift (a : ℕ → ℝ) (q c : ℝ) :
  (q ≠ 1) →
  (∀ n, a (n + 1) = q * a n) →
  (∃ r, ∀ n, (a (n + 1) + c) = r * (a n + c)) →
  c = 0 :=
sorry

end geometric_sequence_shift_l3285_328586


namespace five_student_committee_from_eight_l3285_328546

theorem five_student_committee_from_eight (n k : ℕ) : n = 8 → k = 5 → Nat.choose n k = 56 := by
  sorry

end five_student_committee_from_eight_l3285_328546


namespace eggs_and_cakes_l3285_328557

def dozen : ℕ := 12

def initial_eggs : ℕ := 7 * dozen
def used_eggs : ℕ := 5 * dozen
def eggs_per_cake : ℕ := (3 * dozen) / 2

theorem eggs_and_cakes :
  let remaining_eggs := initial_eggs - used_eggs
  let possible_cakes := remaining_eggs / eggs_per_cake
  remaining_eggs = 24 ∧ possible_cakes = 1 := by sorry

end eggs_and_cakes_l3285_328557


namespace pigeonhole_divisibility_l3285_328549

theorem pigeonhole_divisibility (n : ℕ+) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) := by
  sorry

end pigeonhole_divisibility_l3285_328549


namespace bottles_taken_home_l3285_328544

def bottles_brought : ℕ := 50
def bottles_drunk : ℕ := 38

theorem bottles_taken_home : 
  bottles_brought - bottles_drunk = 12 := by sorry

end bottles_taken_home_l3285_328544


namespace inequality_proof_l3285_328525

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a + c)^2 + (b + d)^2) ≤ Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ∧
  Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ≤ Real.sqrt ((a + c)^2 + (b + d)^2) + (2 * |a * d - b * c|) / Real.sqrt ((a + c)^2 + (b + d)^2) :=
by sorry

end inequality_proof_l3285_328525


namespace quadratic_roots_properties_l3285_328515

/-- The quadratic equation x^2 + 3x - 1 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 1 = 0

/-- The two roots of the quadratic equation -/
noncomputable def root1 : ℝ := sorry
noncomputable def root2 : ℝ := sorry

/-- Proposition p: The two roots have opposite signs -/
def p : Prop := root1 * root2 < 0

/-- Proposition q: The sum of the two roots is 3 -/
def q : Prop := root1 + root2 = 3

theorem quadratic_roots_properties : p ∧ ¬q := by sorry

end quadratic_roots_properties_l3285_328515


namespace whitney_purchase_cost_is_445_62_l3285_328593

/-- Calculates the total cost of Whitney's purchase given the specified conditions --/
def whitneyPurchaseCost : ℝ :=
  let whaleBookCount : ℕ := 15
  let fishBookCount : ℕ := 12
  let sharkBookCount : ℕ := 5
  let magazineCount : ℕ := 8
  let whaleBookPrice : ℝ := 14
  let fishBookPrice : ℝ := 13
  let sharkBookPrice : ℝ := 10
  let magazinePrice : ℝ := 3
  let fishBookDiscount : ℝ := 0.1
  let salesTaxRate : ℝ := 0.05

  let whaleBooksCost := whaleBookCount * whaleBookPrice
  let fishBooksCost := fishBookCount * fishBookPrice * (1 - fishBookDiscount)
  let sharkBooksCost := sharkBookCount * sharkBookPrice
  let magazinesCost := magazineCount * magazinePrice

  let totalBeforeTax := whaleBooksCost + fishBooksCost + sharkBooksCost + magazinesCost
  let salesTax := totalBeforeTax * salesTaxRate
  let totalCost := totalBeforeTax + salesTax

  totalCost

/-- Theorem stating that Whitney's total purchase cost is $445.62 --/
theorem whitney_purchase_cost_is_445_62 : whitneyPurchaseCost = 445.62 := by
  sorry


end whitney_purchase_cost_is_445_62_l3285_328593


namespace lineup_combinations_l3285_328597

def team_size : ℕ := 12
def strong_players : ℕ := 4
def positions_to_fill : ℕ := 5

theorem lineup_combinations : 
  (strong_players * (strong_players - 1) * 
   (team_size - 2) * (team_size - 3) * (team_size - 4)) = 8640 := by
  sorry

end lineup_combinations_l3285_328597


namespace cattle_train_speed_calculation_l3285_328514

/-- The speed of the cattle train in miles per hour -/
def cattle_train_speed : ℝ := 93.33333333333333

/-- The speed of the diesel train in miles per hour -/
def diesel_train_speed (x : ℝ) : ℝ := x - 33

/-- The time difference between the trains' departures in hours -/
def time_difference : ℝ := 6

/-- The travel time of the diesel train in hours -/
def diesel_travel_time : ℝ := 12

/-- The total distance between the trains after the diesel train's travel -/
def total_distance : ℝ := 1284

theorem cattle_train_speed_calculation :
  time_difference * cattle_train_speed +
  diesel_travel_time * cattle_train_speed +
  diesel_travel_time * (diesel_train_speed cattle_train_speed) = total_distance := by
  sorry

end cattle_train_speed_calculation_l3285_328514


namespace complementary_angles_difference_l3285_328550

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- angles are complementary
  a / b = 5 / 4 → -- ratio of angles is 5:4
  |a - b| = 10 := by sorry

end complementary_angles_difference_l3285_328550


namespace third_term_is_negative_45_l3285_328575

/-- A geometric sequence with common ratio -3 and sum of first 2 terms equal to 10 -/
structure GeometricSequence where
  a₁ : ℝ
  ratio : ℝ
  sum_first_two : ℝ
  ratio_eq : ratio = -3
  sum_eq : a₁ + a₁ * ratio = sum_first_two
  sum_first_two_eq : sum_first_two = 10

/-- The third term of the geometric sequence -/
def third_term (seq : GeometricSequence) : ℝ :=
  seq.a₁ * seq.ratio^2

theorem third_term_is_negative_45 (seq : GeometricSequence) :
  third_term seq = -45 := by
  sorry

#check third_term_is_negative_45

end third_term_is_negative_45_l3285_328575


namespace f_properties_l3285_328588

def f (x : ℝ) := x^3 - 6*x + 5

theorem f_properties :
  let sqrt2 := Real.sqrt 2
  ∀ x y : ℝ,
  (∀ x ∈ Set.Ioo (-sqrt2) sqrt2, ∀ y ∈ Set.Ioo (-sqrt2) sqrt2, x < y → f x > f y) ∧
  (∀ x ∈ Set.Iic (-sqrt2), ∀ y ∈ Set.Iic (-sqrt2), x < y → f x < f y) ∧
  (∀ x ∈ Set.Ioi sqrt2, ∀ y ∈ Set.Ioi sqrt2, x < y → f x < f y) ∧
  (f (-sqrt2) = 5 + 4*sqrt2) ∧
  (f sqrt2 = 5 - 4*sqrt2) ∧
  (∀ a : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) ↔
    5 - 4*sqrt2 < a ∧ a < 5 + 4*sqrt2) :=
by sorry

end f_properties_l3285_328588


namespace inequality_proof_l3285_328576

theorem inequality_proof (a b : Real) (ha : 0 < a ∧ a < Real.pi / 2) (hb : 0 < b ∧ b < Real.pi / 2) :
  (5 / Real.cos a ^ 2) + (5 / (Real.sin a ^ 2 * Real.sin b ^ 2 * Real.cos b ^ 2)) ≥ 27 * Real.cos a + 36 * Real.sin a := by
  sorry

end inequality_proof_l3285_328576


namespace total_fishes_caught_l3285_328567

/-- The number of fishes caught by Hazel and her father in Lake Erie -/
theorem total_fishes_caught (hazel_fishes : Nat) (father_fishes : Nat)
  (h1 : hazel_fishes = 48)
  (h2 : father_fishes = 46) :
  hazel_fishes + father_fishes = 94 := by
  sorry

end total_fishes_caught_l3285_328567


namespace sum_two_smallest_trite_numbers_l3285_328533

def is_trite (n : ℕ+) : Prop :=
  ∃ (d : Fin 12 → ℕ+),
    (∀ i j, i < j → d i < d j) ∧
    d 0 = 1 ∧
    d 11 = n ∧
    (∀ k, k ∣ n ↔ ∃ i, d i = k) ∧
    5 + (d 5) * (d 5 + d 3) = (d 6) * (d 3)

theorem sum_two_smallest_trite_numbers : 
  ∃ (a b : ℕ+), is_trite a ∧ is_trite b ∧ 
  (∀ n : ℕ+, is_trite n → a ≤ n) ∧
  (∀ n : ℕ+, is_trite n ∧ n ≠ a → b ≤ n) ∧
  a + b = 151127 :=
sorry

end sum_two_smallest_trite_numbers_l3285_328533


namespace product_of_equal_sums_l3285_328545

theorem product_of_equal_sums (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end product_of_equal_sums_l3285_328545


namespace constant_function_theorem_l3285_328563

def IsNonZeroInteger (x : ℚ) : Prop := ∃ (n : ℤ), n ≠ 0 ∧ x = n

theorem constant_function_theorem (f : ℚ → ℚ) 
  (h : ∀ x y, IsNonZeroInteger x → IsNonZeroInteger y → 
    f ((x + y) / 3) = (f x + f y) / 2) :
  ∃ c, ∀ x, IsNonZeroInteger x → f x = c := by
sorry

end constant_function_theorem_l3285_328563


namespace geometric_sequence_property_l3285_328535

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 4)^2 - 34 * (a 4) + 64 = 0 →
  (a 8)^2 - 34 * (a 8) + 64 = 0 →
  a 6 = 8 := by
  sorry

end geometric_sequence_property_l3285_328535


namespace min_value_of_s_l3285_328511

theorem min_value_of_s (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 3 * x^2 + 2 * y^2 + z^2 = 1) :
  (1 + z) / (x * y * z) ≥ 8 * Real.sqrt 6 := by
  sorry

end min_value_of_s_l3285_328511


namespace proportional_relation_l3285_328569

/-- Given that x is directly proportional to y^4 and y is inversely proportional to z^(1/3),
    prove that if x = 8 when z = 27, then x = 81/32 when z = 64 -/
theorem proportional_relation (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h1 : x = k₁ * y^4)
    (h2 : y = k₂ / z^(1/3))
    (h3 : x = 8 ∧ z = 27) :
    z = 64 → x = 81/32 := by
  sorry

end proportional_relation_l3285_328569


namespace inscribed_sphere_surface_area_l3285_328596

/-- Given a cube with edge length 4, the surface area of its inscribed sphere is 16π. -/
theorem inscribed_sphere_surface_area (edge_length : ℝ) (h : edge_length = 4) :
  let radius : ℝ := edge_length / 2
  4 * π * radius^2 = 16 * π :=
by sorry

end inscribed_sphere_surface_area_l3285_328596


namespace handball_tournament_impossibility_l3285_328558

structure Tournament :=
  (teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

def total_games (t : Tournament) : ℕ :=
  t.teams * (t.teams - 1) / 2

def total_points (t : Tournament) : ℕ :=
  total_games t * (t.points_for_win + t.points_for_loss)

theorem handball_tournament_impossibility 
  (t : Tournament)
  (h1 : t.teams = 14)
  (h2 : t.points_for_win = 2)
  (h3 : t.points_for_draw = 1)
  (h4 : t.points_for_loss = 0)
  (h5 : ∀ (i j : ℕ), i ≠ j → i < t.teams → j < t.teams → 
       ∃ (pi pj : ℕ), pi ≠ pj ∧ pi ≤ total_points t ∧ pj ≤ total_points t) :
  ¬(∃ (top bottom : Finset ℕ), 
    top.card = 3 ∧ 
    bottom.card = 3 ∧ 
    (∀ i ∈ top, ∀ j ∈ bottom, 
      ∃ (pi pj : ℕ), pi > pj ∧ 
      pi ≤ total_points t ∧ 
      pj ≤ total_points t)) :=
sorry

end handball_tournament_impossibility_l3285_328558


namespace dans_remaining_money_dans_remaining_money_proof_l3285_328542

/-- Calculates the remaining money after purchases and tax --/
theorem dans_remaining_money (initial_amount : ℚ) 
  (candy_price : ℚ) (candy_count : ℕ) 
  (gum_price : ℚ) (soda_price : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_before_tax := candy_price * candy_count + gum_price + soda_price
  let total_tax := total_before_tax * tax_rate
  let total_cost := total_before_tax + total_tax
  initial_amount - total_cost

/-- Proves that Dan's remaining money is $40.98 --/
theorem dans_remaining_money_proof :
  dans_remaining_money 50 1.75 3 0.85 2.25 0.08 = 40.98 := by
  sorry

end dans_remaining_money_dans_remaining_money_proof_l3285_328542


namespace circle_line_intersection_range_l3285_328519

/-- Given a circle C and a line l, if they have a common point, 
    then the range of values for a is [-1/2, 1/2) -/
theorem circle_line_intersection_range (a : ℝ) : 
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*a*x + 2*a*y + 2*a^2 + 2*a - 1 = 0}
  let l := {(x, y) : ℝ × ℝ | x - y - 1 = 0}
  (∃ p, p ∈ C ∩ l) → a ∈ Set.Icc (-1/2) (1/2) := by
sorry

end circle_line_intersection_range_l3285_328519


namespace special_hexagon_perimeter_l3285_328517

/-- An equilateral hexagon with specified angle measures and area -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assertion that the hexagon is equilateral
  is_equilateral : True
  -- Assertion about the interior angles
  angle_condition : True
  -- Area of the hexagon
  area : ℝ
  -- The area is 12
  area_is_twelve : area = 12

/-- The perimeter of a SpecialHexagon is 12 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : h.side * 6 = 12 := by
  sorry

#check special_hexagon_perimeter

end special_hexagon_perimeter_l3285_328517


namespace min_bushes_for_zucchinis_l3285_328559

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 11

/-- The number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℕ := 3

/-- The number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 60

/-- The function to calculate the number of bushes needed for a given number of zucchinis -/
def bushes_needed (zucchinis : ℕ) : ℕ :=
  (zucchinis * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush

theorem min_bushes_for_zucchinis :
  bushes_needed target_zucchinis = 17 := by sorry

end min_bushes_for_zucchinis_l3285_328559


namespace pizza_price_correct_l3285_328521

/-- The price of one box of pizza -/
def pizza_price : ℝ := 12

/-- The price of one pack of potato fries -/
def fries_price : ℝ := 0.3

/-- The price of one can of soda -/
def soda_price : ℝ := 2

/-- The number of pizza boxes sold -/
def pizza_sold : ℕ := 15

/-- The number of potato fries packs sold -/
def fries_sold : ℕ := 40

/-- The number of soda cans sold -/
def soda_sold : ℕ := 25

/-- The fundraising goal -/
def goal : ℝ := 500

/-- The amount still needed to reach the goal -/
def amount_needed : ℝ := 258

theorem pizza_price_correct : 
  pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold = goal - amount_needed :=
by sorry

end pizza_price_correct_l3285_328521


namespace amaya_total_marks_l3285_328584

/-- Represents the marks scored in different subjects -/
structure Marks where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total marks across all subjects -/
def total_marks (m : Marks) : ℕ :=
  m.music + m.social_studies + m.arts + m.maths

/-- Theorem stating the total marks Amaya scored -/
theorem amaya_total_marks :
  ∀ (m : Marks),
  m.music = 70 →
  m.social_studies = m.music + 10 →
  m.arts - m.maths = 20 →
  m.maths = (9 : ℕ) * m.arts / 10 →
  total_marks m = 530 := by
  sorry


end amaya_total_marks_l3285_328584


namespace total_rowing_campers_l3285_328574

def morning_rowing : ℕ := 13
def afternoon_rowing : ℕ := 21

theorem total_rowing_campers :
  morning_rowing + afternoon_rowing = 34 := by
  sorry

end total_rowing_campers_l3285_328574


namespace fraction_zero_implies_x_equals_one_l3285_328500

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  x ≠ -1 →
  (x^2 - 1) / (x + 1) = 0 →
  x = 1 := by
sorry

end fraction_zero_implies_x_equals_one_l3285_328500


namespace holly_chocolate_milk_l3285_328572

/-- Holly's chocolate milk consumption throughout the day -/
def chocolate_milk_problem (initial_consumption breakfast_consumption lunch_consumption dinner_consumption new_container_size : ℕ) : Prop :=
  let remaining_milk := new_container_size - (lunch_consumption + dinner_consumption)
  remaining_milk = 48

/-- Theorem stating Holly ends the day with 48 ounces of chocolate milk -/
theorem holly_chocolate_milk :
  chocolate_milk_problem 8 8 8 8 64 := by
  sorry

end holly_chocolate_milk_l3285_328572


namespace smallest_four_digit_multiple_of_18_l3285_328571

theorem smallest_four_digit_multiple_of_18 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 18 = 0 → n ≥ 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l3285_328571


namespace area_of_stacked_squares_l3285_328581

/-- The area of a 24-sided polygon formed by stacking three identical square sheets -/
theorem area_of_stacked_squares (side_length : ℝ) (h : side_length = 8) :
  let diagonal := side_length * Real.sqrt 2
  let radius := diagonal / 2
  let triangle_area := (1/2) * radius^2 * Real.sin (π/6)
  let total_area := 12 * triangle_area
  total_area = 96 := by sorry

end area_of_stacked_squares_l3285_328581


namespace total_staff_is_250_l3285_328538

/-- Represents a hospital with doctors and nurses -/
structure Hospital where
  doctors : ℕ
  nurses : ℕ

/-- The total number of staff (doctors and nurses) in a hospital -/
def Hospital.total (h : Hospital) : ℕ := h.doctors + h.nurses

/-- A hospital satisfying the given conditions -/
def special_hospital : Hospital :=
  { doctors := 100,  -- This is derived from the ratio, not given directly
    nurses := 150 }

theorem total_staff_is_250 :
  (special_hospital.doctors : ℚ) / special_hospital.nurses = 2 / 3 ∧
  special_hospital.nurses = 150 →
  special_hospital.total = 250 := by
  sorry

end total_staff_is_250_l3285_328538


namespace train_time_calculation_l3285_328577

/-- Proves that the additional time for train-related activities is 15.5 minutes --/
theorem train_time_calculation (distance : ℝ) (walk_speed : ℝ) (train_speed : ℝ) 
  (walk_time_difference : ℝ) :
  distance = 1.5 →
  walk_speed = 3 →
  train_speed = 20 →
  walk_time_difference = 10 →
  ∃ (x : ℝ), x = 15.5 ∧ 
    (distance / walk_speed) * 60 = (distance / train_speed) * 60 + x + walk_time_difference :=
by
  sorry

end train_time_calculation_l3285_328577


namespace inverse_proportion_point_ordering_l3285_328594

/-- Given three points A(-3, y₁), B(-2, y₂), C(3, y₃) on the graph of y = -2/x,
    prove that y₃ < y₁ < y₂ -/
theorem inverse_proportion_point_ordering (y₁ y₂ y₃ : ℝ) : 
  y₁ = -2 / (-3) → y₂ = -2 / (-2) → y₃ = -2 / 3 → y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end inverse_proportion_point_ordering_l3285_328594


namespace fourth_month_sale_l3285_328579

theorem fourth_month_sale 
  (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ)
  (h1 : sale1 = 5435)
  (h2 : sale2 = 5927)
  (h3 : sale3 = 5855)
  (h5 : sale5 = 5562)
  (h6 : sale6 = 3991)
  (h_avg : average_sale = 5500)
  : ∃ sale4 : ℕ, sale4 = 6230 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

#check fourth_month_sale

end fourth_month_sale_l3285_328579


namespace track_distance_proof_l3285_328568

/-- The distance Albert needs to run in total, in meters. -/
def total_distance : ℝ := 99

/-- The number of laps Albert has already completed. -/
def completed_laps : ℕ := 6

/-- The number of additional laps Albert will run. -/
def additional_laps : ℕ := 5

/-- The distance around the track, in meters. -/
def track_distance : ℝ := 9

theorem track_distance_proof : 
  (completed_laps + additional_laps : ℝ) * track_distance = total_distance := by
  sorry

end track_distance_proof_l3285_328568


namespace karen_wrong_answers_l3285_328543

/-- Represents the number of wrong answers for each person -/
structure TestResults where
  karen : ℕ
  leo : ℕ
  morgan : ℕ
  nora : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (r : TestResults) : Prop :=
  r.karen + r.leo = r.morgan + r.nora ∧
  r.karen + r.nora = r.leo + r.morgan + 3 ∧
  r.morgan = 6

theorem karen_wrong_answers (r : TestResults) (h : satisfiesConditions r) : r.karen = 6 := by
  sorry

#check karen_wrong_answers

end karen_wrong_answers_l3285_328543


namespace fir_trees_count_l3285_328529

/-- Represents the statements made by each child --/
inductive Statement
  | Anya : Statement
  | Borya : Statement
  | Vera : Statement
  | Gena : Statement

/-- Represents the gender of each child --/
inductive Gender
  | Boy : Gender
  | Girl : Gender

/-- Associates each child with their gender --/
def childGender : Statement → Gender
  | Statement.Anya => Gender.Girl
  | Statement.Borya => Gender.Boy
  | Statement.Vera => Gender.Girl
  | Statement.Gena => Gender.Boy

/-- Checks if a given number satisfies a child's statement --/
def satisfiesStatement (n : ℕ) : Statement → Bool
  | Statement.Anya => n = 15
  | Statement.Borya => n % 11 = 0
  | Statement.Vera => n < 25
  | Statement.Gena => n % 22 = 0

/-- Theorem: The number of fir trees is 11 --/
theorem fir_trees_count : 
  ∃ (n : ℕ) (t₁ t₂ : Statement), 
    n = 11 ∧ 
    childGender t₁ ≠ childGender t₂ ∧
    satisfiesStatement n t₁ ∧ 
    satisfiesStatement n t₂ ∧
    (∀ t : Statement, t ≠ t₁ → t ≠ t₂ → ¬satisfiesStatement n t) :=
  sorry

end fir_trees_count_l3285_328529


namespace hexagon_perimeter_hexagon_perimeter_proof_l3285_328590

/-- The perimeter of a regular hexagon with side length 8 is 48. -/
theorem hexagon_perimeter : ℕ → ℕ
  | 6 => 48
  | _ => 0

#check hexagon_perimeter
-- hexagon_perimeter : ℕ → ℕ

theorem hexagon_perimeter_proof (n : ℕ) (h : n = 6) : 
  hexagon_perimeter n = 8 * n :=
by sorry

end hexagon_perimeter_hexagon_perimeter_proof_l3285_328590


namespace inequalities_proof_l3285_328564

theorem inequalities_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) :=
by sorry

end inequalities_proof_l3285_328564


namespace diane_stamp_arrangements_l3285_328599

/-- Represents a collection of stamps with their quantities -/
def StampCollection := List (Nat × Nat)

/-- Represents an arrangement of stamps -/
def StampArrangement := List Nat

/-- Returns true if the arrangement sums to the target value -/
def isValidArrangement (arrangement : StampArrangement) (target : Nat) : Bool :=
  arrangement.sum = target

/-- Returns true if the arrangement is possible given the stamp collection -/
def isPossibleArrangement (arrangement : StampArrangement) (collection : StampCollection) : Bool :=
  sorry

/-- Counts the number of unique arrangements given a stamp collection and target sum -/
def countUniqueArrangements (collection : StampCollection) (target : Nat) : Nat :=
  sorry

/-- Diane's stamp collection -/
def dianeCollection : StampCollection :=
  [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]

theorem diane_stamp_arrangements :
  countUniqueArrangements dianeCollection 12 = 30 := by sorry

end diane_stamp_arrangements_l3285_328599


namespace city_parking_fee_l3285_328527

def weekly_salary : ℚ := 450
def federal_tax_rate : ℚ := 1/3
def state_tax_rate : ℚ := 8/100
def health_insurance : ℚ := 50
def life_insurance : ℚ := 20
def final_paycheck : ℚ := 184

theorem city_parking_fee :
  let after_federal := weekly_salary * (1 - federal_tax_rate)
  let after_state := after_federal * (1 - state_tax_rate)
  let after_insurance := after_state - health_insurance - life_insurance
  after_insurance - final_paycheck = 22 := by sorry

end city_parking_fee_l3285_328527


namespace cosine_sine_values_l3285_328537

/-- If the sum of cos²ⁿ(θ) from n=0 to infinity equals 9, 
    then cos(2θ) = 7/9 and sin²(θ) = 1/9 -/
theorem cosine_sine_values (θ : ℝ) 
  (h : ∑' n, (Real.cos θ)^(2*n) = 9) : 
  Real.cos (2*θ) = 7/9 ∧ Real.sin θ^2 = 1/9 := by
  sorry

end cosine_sine_values_l3285_328537


namespace defective_pens_l3285_328591

/-- The number of defective pens in a box of 12 pens, given the probability of selecting two non-defective pens. -/
theorem defective_pens (total : ℕ) (prob : ℚ) (h_total : total = 12) (h_prob : prob = 22727272727272727 / 100000000000000000) :
  ∃ (defective : ℕ), defective = 6 ∧ 
    (prob = (↑(total - defective) / ↑total) * (↑(total - defective - 1) / ↑(total - 1))) :=
by sorry

end defective_pens_l3285_328591


namespace no_solution_exists_l3285_328585

theorem no_solution_exists : ¬∃ x : ℝ, 2 < 3 * x ∧ 3 * x < 4 ∧ 1 < 5 * x ∧ 5 * x < 3 := by
  sorry

end no_solution_exists_l3285_328585


namespace total_miles_is_35_l3285_328541

def andrew_daily_miles : ℕ := 2
def peter_extra_miles : ℕ := 3
def days : ℕ := 5

def total_miles : ℕ := 
  (andrew_daily_miles * days) + ((andrew_daily_miles + peter_extra_miles) * days)

theorem total_miles_is_35 : total_miles = 35 := by
  sorry

end total_miles_is_35_l3285_328541


namespace legos_lost_l3285_328556

def initial_legos : ℕ := 2080
def current_legos : ℕ := 2063

theorem legos_lost : initial_legos - current_legos = 17 := by
  sorry

end legos_lost_l3285_328556


namespace expected_occurrences_is_two_l3285_328540

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.2

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.4

/-- The number of trials -/
def num_trials : ℕ := 25

/-- The expected number of trials where both events occur simultaneously -/
def expected_occurrences : ℝ := num_trials * (prob_A * prob_B)

/-- Theorem stating that the expected number of occurrences is 2 -/
theorem expected_occurrences_is_two : expected_occurrences = 2 := by
  sorry

end expected_occurrences_is_two_l3285_328540


namespace shaded_square_covers_all_rows_l3285_328530

def shaded_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => shaded_sequence n + (n + 2)

def covers_all_remainders (n : ℕ) : Prop :=
  ∀ k : Fin 12, ∃ m : ℕ, m ≤ n ∧ shaded_sequence m % 12 = k

theorem shaded_square_covers_all_rows :
  covers_all_remainders 11 ∧ shaded_sequence 11 = 144 ∧
  ∀ k < 11, ¬covers_all_remainders k :=
sorry

end shaded_square_covers_all_rows_l3285_328530


namespace martha_exceptional_savings_l3285_328592

/-- Represents Martha's savings over a week -/
def MarthaSavings (daily_allowance : ℚ) (regular_fraction : ℚ) (exceptional_fraction : ℚ) : ℚ :=
  6 * (daily_allowance * regular_fraction) + (daily_allowance * exceptional_fraction)

/-- Theorem stating the fraction Martha saved on the exceptional day -/
theorem martha_exceptional_savings :
  ∀ (daily_allowance : ℚ),
  daily_allowance = 12 →
  MarthaSavings daily_allowance (1/2) (1/4) = 39 :=
by
  sorry


end martha_exceptional_savings_l3285_328592


namespace bathroom_visit_interval_l3285_328531

/-- Calculates the time between bathroom visits during a movie -/
theorem bathroom_visit_interval (movie_duration : Real) (visit_count : Nat) : 
  movie_duration = 2.5 ∧ visit_count = 3 → 
  (movie_duration * 60) / (visit_count + 1) = 37.5 := by
  sorry

end bathroom_visit_interval_l3285_328531


namespace spencer_total_distance_l3285_328573

/-- The total distance Spencer walked on Saturday -/
def total_distance (house_to_library library_to_post_office post_office_to_house : ℝ) : ℝ :=
  house_to_library + library_to_post_office + post_office_to_house

/-- Theorem stating that Spencer walked 0.8 mile in total -/
theorem spencer_total_distance :
  total_distance 0.3 0.1 0.4 = 0.8 := by
  sorry

end spencer_total_distance_l3285_328573


namespace oblique_view_isosceles_implies_right_trapezoid_l3285_328513

/-- A plane figure. -/
structure PlaneFigure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- An oblique view of a plane figure. -/
structure ObliqueView where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents that a figure is an isosceles trapezoid. -/
def is_isosceles_trapezoid (f : ObliqueView) : Prop :=
  sorry

/-- Represents the base angle of a figure. -/
def base_angle (f : ObliqueView) : ℝ :=
  sorry

/-- Represents that a figure is a right trapezoid. -/
def is_right_trapezoid (f : PlaneFigure) : Prop :=
  sorry

/-- 
If the oblique view of a plane figure is an isosceles trapezoid 
with a base angle of 45°, then the original figure is a right trapezoid.
-/
theorem oblique_view_isosceles_implies_right_trapezoid 
  (f : PlaneFigure) (v : ObliqueView) :
  is_isosceles_trapezoid v → base_angle v = 45 → is_right_trapezoid f :=
by
  sorry

end oblique_view_isosceles_implies_right_trapezoid_l3285_328513


namespace arithmetic_mean_of_special_set_l3285_328566

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let s : Finset ℕ := Finset.range n
  let f : ℕ → ℝ := λ i => if i = 0 then 1/n + 2/n^2 else 1
  (s.sum f) / n = 1 + 2/n^2 := by
  sorry

end arithmetic_mean_of_special_set_l3285_328566


namespace line_symmetry_l3285_328502

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Define the axis of symmetry
def axis_of_symmetry (x : ℝ) : Prop := x = 1

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := 2 * x + y - 8 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_line x₀ y₀ ∧ axis_of_symmetry ((x + x₀) / 2)) →
  symmetric_line x y :=
sorry

end line_symmetry_l3285_328502


namespace power_of_product_l3285_328518

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  sorry

end power_of_product_l3285_328518


namespace age_sum_proof_l3285_328509

theorem age_sum_proof (A B C : ℕ) 
  (h1 : A = B + C + 16) 
  (h2 : A^2 = (B + C)^2 + 1632) : 
  A + B + C = 102 := by
sorry

end age_sum_proof_l3285_328509


namespace age_solution_l3285_328526

/-- The ages of Petya and Anya satisfy the given conditions -/
def age_relationship (petya_age anya_age : ℕ) : Prop :=
  (petya_age = 3 * anya_age) ∧ (petya_age - anya_age = 8)

/-- Theorem stating that Petya is 12 years old and Anya is 4 years old -/
theorem age_solution : ∃ (petya_age anya_age : ℕ), 
  age_relationship petya_age anya_age ∧ petya_age = 12 ∧ anya_age = 4 := by
  sorry

end age_solution_l3285_328526


namespace even_sequence_sum_l3285_328578

theorem even_sequence_sum (n : ℕ) (sum : ℕ) : sum = n * (n + 1) → 2 * n = 30 :=
  sorry

#check even_sequence_sum

end even_sequence_sum_l3285_328578


namespace minimum_travel_time_l3285_328523

/-- The minimum time for a person to travel from point A to point B -/
theorem minimum_travel_time (BC : ℝ) (angle_BAC : ℝ) (swimming_speed : ℝ) 
  (h1 : BC = 30)
  (h2 : angle_BAC = 15 * π / 180)
  (h3 : swimming_speed = 3) :
  ∃ t : ℝ, t = 20 ∧ 
  ∀ t' : ℝ, t' ≥ t ∧ 
  ∃ d : ℝ, t' = d / (swimming_speed * Real.sqrt 2) + Real.sqrt (d^2 - BC^2) / swimming_speed :=
by sorry

end minimum_travel_time_l3285_328523


namespace adult_meals_count_l3285_328589

/-- The number of meals that can feed children -/
def childMeals : ℕ := 90

/-- The number of adults who have their meal -/
def adultsMealed : ℕ := 35

/-- The number of children that can be fed with remaining food after some adults eat -/
def remainingChildMeals : ℕ := 45

/-- The number of meals initially available for adults -/
def adultMeals : ℕ := 80

theorem adult_meals_count :
  adultMeals = childMeals - remainingChildMeals + adultsMealed :=
by sorry

end adult_meals_count_l3285_328589


namespace partitioned_square_theorem_main_theorem_l3285_328555

/-- A square with interior points and partitioned into triangles -/
structure PartitionedSquare where
  /-- The number of interior points in the square -/
  num_interior_points : ℕ
  /-- The number of line segments drawn -/
  num_segments : ℕ
  /-- The number of triangles formed -/
  num_triangles : ℕ
  /-- Ensures that the number of interior points is 1965 -/
  h_points : num_interior_points = 1965

/-- Theorem stating the relationship between the number of interior points,
    line segments, and triangles in a partitioned square -/
theorem partitioned_square_theorem (ps : PartitionedSquare) :
  ps.num_segments = 5896 ∧ ps.num_triangles = 3932 := by
  sorry

/-- Main theorem proving the specific case for 1965 interior points -/
theorem main_theorem : 
  ∃ ps : PartitionedSquare, ps.num_segments = 5896 ∧ ps.num_triangles = 3932 := by
  sorry

end partitioned_square_theorem_main_theorem_l3285_328555


namespace probability_of_marked_items_l3285_328553

theorem probability_of_marked_items 
  (N M n m : ℕ) 
  (h1 : M ≤ N) 
  (h2 : n ≤ N) 
  (h3 : m ≤ n) 
  (h4 : m ≤ M) :
  (Nat.choose M m * Nat.choose (N - M) (n - m)) / Nat.choose N n = 
  (Nat.choose M m * Nat.choose (N - M) (n - m)) / Nat.choose N n :=
by sorry

end probability_of_marked_items_l3285_328553


namespace lisa_quiz_goal_l3285_328554

theorem lisa_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (completed_as : ℕ) 
  (h1 : total_quizzes = 40)
  (h2 : goal_percentage = 9/10)
  (h3 : completed_quizzes = 25)
  (h4 : completed_as = 20) : 
  (total_quizzes - completed_quizzes : ℤ) - 
  (↑(total_quizzes * goal_percentage.num) / goal_percentage.den - completed_as : ℚ) = 0 :=
by sorry

end lisa_quiz_goal_l3285_328554


namespace average_of_three_numbers_l3285_328528

theorem average_of_three_numbers (x : ℝ) : 
  (12 + 21 + x) / 3 = 18 → x = 21 := by
  sorry

end average_of_three_numbers_l3285_328528


namespace circle_circumference_after_scaling_l3285_328512

theorem circle_circumference_after_scaling (a b : ℝ) (h1 : a = 7) (h2 : b = 24) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_d := 1.5 * d
  let new_circumference := π * new_d
  new_circumference = 37.5 * π := by
  sorry

end circle_circumference_after_scaling_l3285_328512


namespace inequality_proof_l3285_328507

theorem inequality_proof (a b c : ℝ) (h1 : b > a) (h2 : a > 0) :
  a^2 < b^2 ∧ a*b < b^2 := by
  sorry

end inequality_proof_l3285_328507


namespace pats_running_speed_l3285_328501

/-- Proves that given a 20-mile course, where a person bicycles at 30 mph for 12 minutes
and then runs the rest of the distance, taking a total of 117 minutes to complete the course,
the person's average running speed is 8 mph. -/
theorem pats_running_speed (total_distance : ℝ) (bicycle_speed : ℝ) (bicycle_time : ℝ) (total_time : ℝ)
  (h1 : total_distance = 20)
  (h2 : bicycle_speed = 30)
  (h3 : bicycle_time = 12 / 60)
  (h4 : total_time = 117 / 60) :
  let bicycle_distance := bicycle_speed * bicycle_time
  let run_distance := total_distance - bicycle_distance
  let run_time := total_time - bicycle_time
  run_distance / run_time = 8 := by sorry

end pats_running_speed_l3285_328501


namespace min_value_of_z_l3285_328520

/-- The objective function to be minimized -/
def z (x y : ℝ) : ℝ := 2*x + 5*y

/-- The feasible region defined by the given constraints -/
def feasible_region (x y : ℝ) : Prop :=
  x - y + 2 ≥ 0 ∧ 2*x + 3*y - 6 ≥ 0 ∧ 3*x + 2*y - 9 ≤ 0

/-- Theorem stating that the minimum value of z in the feasible region is 6 -/
theorem min_value_of_z : 
  ∀ x y : ℝ, feasible_region x y → z x y ≥ 6 ∧ ∃ x₀ y₀ : ℝ, feasible_region x₀ y₀ ∧ z x₀ y₀ = 6 :=
sorry

end min_value_of_z_l3285_328520


namespace positive_distinct_solutions_l3285_328561

/-- Given a system of equations, prove the necessary and sufficient conditions for positive and distinct solutions -/
theorem positive_distinct_solutions (a b x y z : ℝ) :
  x + y + z = a →
  x^2 + y^2 + z^2 = b^2 →
  x * y = z^2 →
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ (a > 0 ∧ b^2 < a^2 ∧ a^2 < 3 * b^2) :=
by sorry

end positive_distinct_solutions_l3285_328561


namespace study_supplies_cost_l3285_328562

/-- The cost of study supplies -/
theorem study_supplies_cost 
  (x y z : ℚ) -- x: cost of a pencil, y: cost of an exercise book, z: cost of a ballpoint pen
  (h1 : 3*x + 7*y + z = 3.15) -- First condition
  (h2 : 4*x + 10*y + z = 4.2) -- Second condition
  : x + y + z = 1.05 := by sorry

end study_supplies_cost_l3285_328562


namespace product_sum_in_base_l3285_328583

/-- Converts a number from base b to base 10 -/
def to_base_10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base b -/
def from_base_10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if a number is expressed in a given base -/
def is_in_base (n : ℕ) (b : ℕ) : Prop := sorry

theorem product_sum_in_base (b : ℕ) :
  (b > 1) →
  (is_in_base 13 b) →
  (is_in_base 14 b) →
  (is_in_base 17 b) →
  (is_in_base 5167 b) →
  (to_base_10 13 b * to_base_10 14 b * to_base_10 17 b = to_base_10 5167 b) →
  (from_base_10 (to_base_10 13 b + to_base_10 14 b + to_base_10 17 b) 7 = 50) :=
by sorry

end product_sum_in_base_l3285_328583


namespace waiter_income_fraction_l3285_328565

theorem waiter_income_fraction (salary tips income : ℚ) : 
  income = salary + tips → 
  tips = (5 : ℚ) / 3 * salary → 
  tips / income = (5 : ℚ) / 8 := by
sorry

end waiter_income_fraction_l3285_328565


namespace union_of_A_and_I_minus_B_l3285_328580

def I : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 1, 2}

theorem union_of_A_and_I_minus_B : A ∪ (I \ B) = {0, 1, 2} := by sorry

end union_of_A_and_I_minus_B_l3285_328580


namespace kelly_baking_powder_l3285_328522

/-- The amount of baking powder Kelly has today in boxes -/
def today_amount : ℝ := 0.3

/-- The difference in baking powder between yesterday and today in boxes -/
def difference : ℝ := 0.1

/-- The amount of baking powder Kelly had yesterday in boxes -/
def yesterday_amount : ℝ := today_amount + difference

theorem kelly_baking_powder : yesterday_amount = 0.4 := by
  sorry

end kelly_baking_powder_l3285_328522


namespace marin_apples_l3285_328560

theorem marin_apples (donald_apples : ℕ) (total_apples : ℕ) 
  (h1 : donald_apples = 2)
  (h2 : total_apples = 11) :
  ∃ marin_apples : ℕ, marin_apples + donald_apples = total_apples ∧ marin_apples = 9 := by
  sorry

end marin_apples_l3285_328560
